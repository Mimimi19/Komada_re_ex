from scipy.special import erf

def main(x, a, b1, b2, kappa, ka):
    """
    非線形性 NLNK(x) を定義します（式14）。
    x: フィルターされた刺激 g(t) の値
    a, b1, b2: 非線形性のパラメータ
    kappa: フィルターされた刺激 g(t) の分散 [7]
    ka: 活性化率定数（動力学ブロックのパラメータ）
    """
    return a ** (erf(kappa * x + b1) + 1 ) * ka ** (- 1) + b2

if __name__ == "__main__":
    # テスト用のパラメータ
    a = 1.0