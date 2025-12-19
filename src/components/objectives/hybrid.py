# src/components/objectives/hybrid.py
import numpy as np
from scipy.stats import pearsonr, spearmanr 
from scipy.signal import butter, filtfilt

def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """シンプルな移動平均ローパス。
    win <= 1 の場合は入力をそのまま返す。
    """
    if win is None or win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, kernel, mode="same")

def _bandpass_zero_phase(x: np.ndarray, dt: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """
    ゼロ位相バンドパス（butter + filtfilt）
    dt が無い/不正な場合や、設計不能な場合はそのまま返す。
    """
    if dt is None or dt <= 0:
        return x
    fs = 1.0 / dt
    nyq = 0.5 * fs

    # 正規化周波数（0<low<high<nyq）
    low = low_hz / nyq
    high = high_hz / nyq
    if not (0 < low < high < 1.0):
        return x

    # filtfilt が短すぎる系列で落ちるのを避ける（ざっくり安全策）
    if len(x) < max(32, 3 * (order + 1)):
        return x

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def calculate(output_eval, model_eval, dt=None,
              lp_sec=0.125,   # 低域抽出用のローパス窓(秒)
              w_low=1.0,     # 低周波RMSEの重み
              w_high=3.0,    # 高周波RMSEの重み
              w_corr=0.5,    # 相関の重み（従来の 0.5 を踏襲）
              use_diff_hp=False, # True: 差分で高域を見る（より攻める）
            #   4–30Hz band
              use_band_spearman=True,
              bp_low_hz=4.0,
              bp_high_hz=30.0,
              bp_order=4,
              w_band=1.0
              ):
    """
    RMSEとピアソン相関を組み合わせたハイブリッドスコア
    分布の仮定が不要なRMSEで「値」を合わせ、相関で「形」を合わせる。

    正規分布かどうかに依存しない最強の指標は RMSE（二乗平均平方根誤差） です。これは単純に「正解との距離」を測るため、分布の仮定は不要です。
    ただし、RMSEだけだと「波形のタイミング」より「全体の値のズレ」を優先しすぎることがあります。
    そのため、「RMSE（値のズレを直す）」＋「相関係数（タイミングを合わせる）」を組み合わせるのが、波形フィッティングでは定石です。分布が心配であれば、RMSEの比重を高めれば安全です。

    まじで予測結果が直線になっちゃうので
    従来のRMSEは「大局（低周波）」が合うだけでもスコアが改善しやすく、
    “細かい周波数成分（高周波）” が無視されがちです。
    そこで、ローパスで低域を分離し、高域（残差）にもRMSEを課すことで
    「大局 + 細部」の両方を当てにいきます。
    """
    # 平坦な波形のチェック
    if np.std(model_eval) > 1e-9 and np.std(output_eval) > 1e-9:
        # 1. RMSE (Root Mean Squared Error): 値の絶対的な誤差
        # 小さいほど良い (0に近いほど良い)

        # --- (A) 低周波/高周波に分解 ---
        if dt is not None and dt > 0:
            win = max(3, int(lp_sec / dt))

        else:
            # dtがない場合は安全に「長さに応じた窓」にする（過度に大きくしない）
            win = max(3, int(0.01 * len(output_eval)))  # 長さの1%程度

        low_o = _moving_average(output_eval, win)
        low_m = _moving_average(model_eval, win)

        # 高周波（残差）
        high_o = output_eval - low_o
        high_m = model_eval - low_m

        # さらに“細部”を強く当てたい場合は差分を見る（より高域寄り）
        if use_diff_hp:
            high_o = np.diff(output_eval)
            high_m = np.diff(model_eval)

        # 低域RMSE
        mse_low = np.mean((low_o - low_m) ** 2)
        rmse_low = np.sqrt(mse_low)

        # 高域RMSE
        mse_high = np.mean((high_o - high_m) ** 2)
        rmse_high = np.sqrt(mse_high)

        # 2. Pearson Correlation: 形の類似度
        # 線形な関係性を見るため、正規分布でなくとも「波形の類似度」の指標としては機能する
        r_val, _ = pearsonr(output_eval, model_eval)
        
        band_penalty = 0.0
        if use_band_spearman and (dt is not None) and (dt > 0):
            # 4–30Hz に通して、順位一致を強制
            o_bp = _bandpass_zero_phase(output_eval.astype(np.float64), dt, bp_low_hz, bp_high_hz, order=bp_order)
            m_bp = _bandpass_zero_phase(model_eval.astype(np.float64), dt, bp_low_hz, bp_high_hz, order=bp_order)

            # 定数系列だと spearmanr が nan になるので回避
            if np.std(o_bp) > 1e-9 and np.std(m_bp) > 1e-9:
                rho_band, _ = spearmanr(o_bp, m_bp)
                if np.isfinite(rho_band):
                    # 最小化なので (1 - rho) を足す
                    band_penalty = w_band * (1.0 - float(rho_band))
                else:
                    band_penalty = w_band * 1.0
            else:
                band_penalty = w_band * 1.0

        # --- 目的関数 ---
        # score = (低域RMSE) + (高域RMSE) - (重み * 相関)
        # 相関が1に近づくほどスコアが下がり、RMSEが0に近づくほどスコアが下がる

        score = (w_low * rmse_low) + (w_high * rmse_high) - (w_corr * r_val) + band_penalty
        

    else:
        score = 10.0 # 強いペナルティ

    return score
