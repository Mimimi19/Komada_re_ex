import numpy as np
from scipy.stats import pearsonr

def calculate(output_eval, model_eval):
    """
    RMSEとピアソン相関を組み合わせたハイブリッドスコア
    分布の仮定が不要なRMSEで「値」を合わせ、相関で「形」を合わせる。
    
    正規分布かどうかに依存しない最強の指標は RMSE（二乗平均平方根誤差） です。これは単純に「正解との距離」を測るため、分布の仮定は不要です。
    ただし、RMSEだけだと「波形のタイミング」より「全体の値のズレ」を優先しすぎることがあります。
    そのため、「RMSE（値のズレを直す）」＋「相関係数（タイミングを合わせる）」を組み合わせるのが、波形フィッティングでは定石です。分布が心配であれば、RMSEの比重を高めれば安全です。
    """
    # 平坦な波形のチェック
    if np.std(model_eval) > 1e-9 and np.std(output_eval) > 1e-9:
        # 1. RMSE (Root Mean Squared Error): 値の絶対的な誤差
        # 小さいほど良い (0に近いほど良い)
        mse = np.mean((output_eval - model_eval) ** 2)
        rmse = np.sqrt(mse)
        
        # 2. Pearson Correlation: 形の類似度
        # 線形な関係性を見るため、正規分布でなくとも「波形の類似度」の指標としては機能する
        r_val, _ = pearsonr(output_eval, model_eval)
        
        # --- 目的関数 ---
        # score = RMSE - (重み * 相関)
        # 相関が1に近づくほどスコアが下がり、RMSEが0に近づくほどスコアが下がる
        
        # データのスケール感によりますが、RMSEを重視する設定
        # 例: RMSEが0.1減ることと、相関が0.1上がることが等価になるように調整
        score = rmse - (0.5 * r_val)
        
    else:
        score = 10.0 # 強いペナルティ
        
    return score