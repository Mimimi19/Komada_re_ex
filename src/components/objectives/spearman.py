import numpy as np
from scipy.stats import spearmanr

def calculate(output_eval, model_eval):
    """
    スピアマン順位相関を用いたスコア計算
    戻り値: 最小化したいコスト (相関が高いほど小さい値になる)
    """
    # 平坦な波形のチェック (標準偏差が極小ならペナルティ)
    if np.std(model_eval) > 1e-9 and np.std(output_eval) > 1e-9:
        corr_val, _ = spearmanr(output_eval, model_eval)
        # 相関(Max 1.0) を最小化問題にするため -1 を掛ける
        score = -1.0 * corr_val
    else:
        score = 2.0 # ペナルティ
        
    return score