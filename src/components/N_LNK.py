
import numpy as np

# シグモイド関数を使う
def main(g, a, b1, b2):

    """
    非線形フィルターを適用します（式13）。
    g: 入力信号
    a: シグモイド関数の傾き
    b1, b2: シグモイド関数のパラメータ
    """
    
    # シグモイド関数の計算
    # 非線形フィルターの出力
    nonlinear_output =  a / (1.0 + np.exp(-b1 * (g - b2)))
    
    return nonlinear_output

if __name__ == "__main__":
    # テスト用のパラメータ
    a = 1.0
    b1 = 0.0
    b2 = 0.0