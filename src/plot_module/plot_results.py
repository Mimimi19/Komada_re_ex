# src/plot_module/plot_results.py
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.stats import spearmanr
import glob
import re

# 日本語フォント対応
try:
    import japanize_matplotlib
except ImportError:
    pass

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    import components.L_LNK as L_LNK
    import components.N_LNK as N_LNK
    import components.K_baccus as K_LNK
except ImportError:
    print("エラー: 'src/components' が見つかりません。")
    sys.exit(1)

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_data(config_path):
    """データをロードし、学習時と同じ正規化を行う"""
    cfg = load_yaml(config_path)
    input_path = cfg['input_file']
    output_path = cfg['output_file']
    dt = cfg['dt']

    print(f"Loading data from: {input_path}")
    raw_input = np.genfromtxt(input_path)
    raw_output = np.genfromtxt(output_path)

    # 1. Input: Z-score 正規化
    input_std = np.std(raw_input)
    if input_std > 1e-9:
        norm_input = (raw_input - np.mean(raw_input)) / input_std
    else:
        norm_input = raw_input - np.mean(raw_input)

    # 2. Output: Max-Abs Scaling（※学習側がMaxAbsならこれでOK。学習側がz-scoreなら合わせてください）
    max_val = np.max(np.abs(raw_output))
    if max_val > 1e-9:
        norm_output = raw_output / max_val
    else:
        norm_output = raw_output

    return norm_input, norm_output, dt

def load_params_from_dir(results_dir):
    """パラメータ読み込み関数 (epochファイルを優先)"""
    params = {}

    # J個のalphasの後に続くスカラーパラメータのリスト（Wなし）
    keys_order = [
        'delta', 'a', 'kappa', 'b1', 'b2',
        'ka', 'kfi', 'kfr', 'ksi', 'ksr'
    ]
    num_scalar_params = len(keys_order)

    # 1. まず epoch_*.txt (まとまったファイル) を探す
    epoch_files = glob.glob(os.path.join(results_dir, "epochs", "epoch_*_params.txt"))
    if not epoch_files:
        epoch_files = glob.glob(os.path.join(results_dir, "epoch_*_params.txt"))

    if epoch_files:
        def extract_epoch_num(path):
            match = re.search(r'epoch_(\d+)_params', path)
            return int(match.group(1)) if match else 0

        latest_file = max(epoch_files, key=extract_epoch_num)
        print(f"Loading from single file (PRIORITY): {latest_file}")
        vals = np.loadtxt(latest_file)

        if len(vals) < num_scalar_params:
            print("エラー: パラメータファイルが短すぎます。")
            sys.exit(1)

        # Jの推定: 全パラメータ数 - スカラーパラメータ数
        J = len(vals) - num_scalar_params
        params['J'] = J
        params['alphas'] = vals[0:J]

        for i, k in enumerate(keys_order):
            val = float(vals[J + i])
            # kaのゼロ除算対策
            if k == 'ka' and abs(val) < 1e-9:
                val = 1.0
            params[k] = val

    else:
        # epochファイルがない場合のみ、個別ファイル (L*.txt) を探す
        l_files = glob.glob(os.path.join(results_dir, "L*.txt"))

        if l_files:
            l_files.sort(key=lambda x: int(os.path.basename(x).replace('L', '').replace('.txt', '')))
            alphas = [float(np.loadtxt(f)) for f in l_files]
            params['alphas'] = np.array(alphas)
            params['J'] = len(alphas)

            for k in keys_order:
                p_path = os.path.join(results_dir, f"{k}.txt")
                if os.path.exists(p_path):
                    val = float(np.loadtxt(p_path))
                    if k == 'ka' and abs(val) < 1e-9:
                        print(f"警告: {k} が 0.0 に近いため 1.0 に補正します。")
                        val = 1.0
                    params[k] = val
                else:
                    if k == 'ka':
                        params[k] = 1.0
                    elif k == 'kappa':
                        params[k] = 1.0
                    else:
                        params[k] = 0.0
            print(f"Loaded from individual files. J={params['J']}")
        else:
            print("エラー: パラメータファイルが見つかりません。")
            sys.exit(1)

    return params

def calculate_steady_state(params, use_I2=True):
    """
    パラメータから平均入力に対する定常状態(R, A, I1, I2)を計算する。
    BaccusModel.py の _calculate_steady_state と同じロジック想定。
    """
    a = params['a']
    kappa = params['kappa']
    b1 = params['b1']
    b2 = params['b2']
    ka = params['ka']
    kfi = params['kfi']
    kfr = params['kfr']
    ksi = params['ksi'] if use_I2 else 0.0
    ksr = params['ksr'] if use_I2 else 0.0

    # 1. 定常入力 u_steady の計算 (入力=0 のときの非線形応答)
    u_steady = N_LNK.main(np.array([0.0]), a, kappa, b1, b2, ka)[0]
    u_steady = max(0.0, u_steady)

    # 2. 定常状態の代数的計算
    A_ratio = (ka * u_steady) / kfi if kfi > 1e-9 else 0.0
    I1_ratio = (ka * u_steady) / kfr if kfr > 1e-9 else 0.0

    if use_I2 and (kfr > 1e-9 and ksr > 1e-9):
        # I2 = (ksi * ka) / (kfr * ksr)
        I2_ratio = (ksi * ka) / (kfr * ksr)
    else:
        I2_ratio = 0.0

    # 3. 合計が1になるように正規化
    total = 1.0 + A_ratio + I1_ratio + I2_ratio
    return (1.0/total), (A_ratio/total), (I1_ratio/total), (I2_ratio/total)

def _crop_conv(full, shift_idx, target_len):
    """畳み込み結果 full から shift_idx 以降を target_len で切り出す（足りない場合はゼロ埋め）"""
    if len(full) > shift_idx + target_len:
        return full[shift_idx:shift_idx + target_len]
    out = np.zeros(target_len, dtype=np.float64)
    take = max(0, min(len(full) - shift_idx, target_len))
    if take > 0:
        out[:take] = full[shift_idx:shift_idx + take]
    return out

def run_prediction(params, Input, dt, tau=1.0, tau_short=None, use_I2=True):
    """
    モデルを実行し、g_t, u_t, A_state（最終出力）を返す
    ※案A: Wは使わない
    """
    # 初期状態をパラメータから計算
    R0, A0, I10, I20 = calculate_steady_state(params, use_I2=use_I2)

    alphas = params['alphas']
    J = len(alphas)
    delta = params['delta']

    # --- (A) 長窓 g_long ---
    filter_points = int(tau / dt) + 1
    kernel_long, _ = L_LNK.main(alphas, delta, filter_points, dt, tau)
    g_full = fftconvolve(Input, kernel_long, mode='full')
    shift_idx = int(tau / dt)
    g_long = _crop_conv(g_full, shift_idx, len(Input))

    g_long_std = np.std(g_long)
    if g_long_std > 1e-9:
        g_long = g_long / g_long_std

    # --- (B) 短窓 g_short（任意） ---
    if tau_short is not None and tau_short > 0:
        filter_points_s = int(tau_short / dt) + 1

        # ★重要：短窓は点数が少ないので、基底数を制限（shapeエラー回避）
        J_eff_s = min(J, filter_points_s)
        alphas_s = alphas[:J_eff_s]

        kernel_short, _ = L_LNK.main(alphas_s, delta, filter_points_s, dt, tau_short)
        g_full_s = fftconvolve(Input, kernel_short, mode='full')
        shift_idx_s = int(tau_short / dt)
        g_short = _crop_conv(g_full_s, shift_idx_s, len(Input))

        g_short_std = np.std(g_short)
        if g_short_std > 1e-9:
            g_short = g_short / g_short_std

        # 合成（学習側と同じ：単純和）
        g_t = g_long + g_short
    else:
        g_t = g_long

    # これがないと g_t が大きくなり、非線形関数が飽和することがある
    g_std = np.std(g_t)
    if g_std > 1e-9:
        g_t = g_t / g_std

    # 2. Nonlinear Module
    u_t = N_LNK.main(
        g_t,
        params['a'],
        params['kappa'],
        params['b1'],
        params['b2'],
        params['ka']
    )

    # 3. Kinetic Model（4状態）
    R, A, I1, I2, check = K_LNK.main(
        len(u_t), u_t, dt,
        R0, A0, I10, I20,
        params['ka'], params['kfi'], params['kfr'], params['ksi'] if use_I2 else 0.0, params['ksr'] if use_I2 else 0.0,
        label="Validation"
    )

    return g_t, u_t, A, check

def main(data_config_path, results_dir, tau=1.0):
    if not os.path.exists(results_dir):
        print(f"エラー: ディレクトリが見つかりません: {results_dir}")
        return

    # データロード
    Input, Output_exp, dt = load_data(data_config_path)

    # 学習時の tau_short/use_I2 を config.yaml から読む（無ければ安全なデフォルト）
    train_cfg = load_yaml("config/config.yaml")  # プロジェクトルート想定。必要ならパスを調整
    hp = train_cfg.get("hyper_params", {})
    tau_short = hp.get("tau_short", None)
    use_I2 = bool(hp.get("use_I2", True))

    # パラメータロード
    print(f"Loading parameters from {results_dir}...")
    try:
        params = load_params_from_dir(results_dir)
    except Exception as e:
        print(f"パラメータ読み込みエラー: {e}")
        return

    # 長さ揃え
    min_len = min(len(Input), len(Output_exp))
    Input = Input[:min_len]
    Output_exp = Output_exp[:min_len]

    print("Running model prediction...")
    g_t, u_t, A_raw, check = run_prediction(params, Input, dt, tau=tau, tau_short=tau_short, use_I2=use_I2)

    if check != 1:
        print("Kineticモデル計算失敗。")
        A_raw = np.zeros_like(Output_exp)
        corr = 0.0

    # 長さ揃え
    min_len_eval = min(len(Output_exp), len(A_raw), len(g_t), len(u_t))
    Input = Input[:min_len_eval]
    Output_exp = Output_exp[:min_len_eval]
    A_raw = A_raw[:min_len_eval]
    g_t = g_t[:min_len_eval]
    u_t = u_t[:min_len_eval]

    # predict.txt（相関計算用の正規化後データ）
    Prediction = A_raw.copy()
    Prediction = Prediction - np.mean(Prediction)
    p_max = np.max(np.abs(Prediction))
    if p_max > 1e-9:
        Prediction = Prediction / p_max

    # masked_pre.txt (マスク済み)
    masked_pre = Prediction.copy()
    mask_steps = int(1.0 / dt)  # 1秒マスク
    mask_steps = min(mask_steps, len(masked_pre))

    if len(masked_pre) > mask_steps:
        fill_val = np.mean(masked_pre[mask_steps:])
    else:
        fill_val = 0.0
    masked_pre[:mask_steps] = fill_val

    # 相関計算 (マスク済みデータで計算)
    if check == 1:
        if len(Output_exp) > mask_steps:
            corr, _ = spearmanr(Output_exp[mask_steps:], masked_pre[mask_steps:])
        else:
            corr, _ = spearmanr(Output_exp, masked_pre)
        print(f"Spearman Correlation (Masked): {corr:.6f}")
    else:
        corr = 0.0

    # 保存
    save_dir = os.path.join(results_dir, "validation")
    os.makedirs(save_dir, exist_ok=True)

    np.savetxt(os.path.join(save_dir, "g.txt"), g_t, fmt='%.6f')
    np.savetxt(os.path.join(save_dir, "u.txt"), u_t, fmt='%.6f')
    np.savetxt(os.path.join(save_dir, "predict.txt"), Prediction, fmt='%.6f')
    np.savetxt(os.path.join(save_dir, "masked_pre.txt"), masked_pre, fmt='%.6f')
    print(f"保存完了: {save_dir}")

    # プロット
    total_time = len(Input) * dt
    time_axis = np.arange(len(Input)) * dt
    major_ticks = np.arange(0, int(total_time) + 1, 1)

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=False)

    def setup_axis(ax, title, ylabel):
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, total_time)
        ax.set_xticks(major_ticks)
        ax.grid(which='major', alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=10)

    axes[0].plot(time_axis, Input, color='gray', linewidth=1)
    setup_axis(axes[0], '1. Input Stimulus (Normalized)', 'Contrast')

    axes[1].plot(time_axis, g_t, color='blue', linewidth=1)
    setup_axis(axes[1], '2. Linear Filter Output g(t) [Normalized]', 'Filtered Signal')

    axes[2].plot(time_axis, u_t, color='green', linewidth=1)
    setup_axis(axes[2], '3. Nonlinear Output u(t)', 'Rate (u)')

    # 4段目: 実測値 vs 予測値（案A: A_state）
    axes[3].plot(time_axis, Output_exp, color='black', alpha=0.5, label='Experiment', linewidth=1.5)
    axes[3].plot(time_axis, masked_pre, color='red', alpha=0.8, label=f'Model A_state (Corr={corr:.3f})', linewidth=1.5)
    setup_axis(axes[3], '4. Final Response (A_state)', 'Response')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "detailed_analysis_Astate.png"))
    print("グラフ保存完了")

if __name__ == "__main__":
    # =================================================================
    # ▼▼▼ ここに使用するデータと結果フォルダの詳細を入力してください ▼▼▼
    # =================================================================

    TARGET_CONFIG = "config/data/ret2p-1.yaml"
    TARGET_DIR = "scripts/ret2p/20251219_1303"
    TARGET_TAU = 1.0

    # =================================================================

    if len(sys.argv) > 1:
        TARGET_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        TARGET_CONFIG = sys.argv[2]

    print(f"--- 実行設定 ---")
    print(f"Config : {TARGET_CONFIG}")
    print(f"Results: {TARGET_DIR}")
    print(f"Tau    : {TARGET_TAU}")
    print(f"---------------")

    main(TARGET_CONFIG, TARGET_DIR, TARGET_TAU)
