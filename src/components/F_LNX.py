import numpy as np
import matplotlib.pyplot as plt

import components.BasisFunctions as BasisFunctions


def main(alphas, delta, dt, tau, j, J=15):
    """
    FLNKフィルターカーネルを生成します（式12）。
    alphas: 各基底関数の重み係数リスト
    delta: 遅延パラメータ (s)
    dt: タイムステップ (s)
    j: 使用する基底関数の数
    """

    t_values = np.arange(delta, tau, dt)

    # FLNKフィルターカーネルの生成
    kernel = np.zeros_like(t_values)  # カーネルの初期化

    for j in range(J):
        shifted_t = t_values + delta  # 遅延補正
        kernel += alphas[j] * BasisFunctions.main(shifted_t, j + 1, tau)
    # 畳み込みようにフィルターを反転
    return kernel[::-1], t_values[::-1]

if __name__ == "__main__":
    alphas = [0.1 * (i + 1) for i in range(J)] # 重み係数の例
    delta = 0.5  # 遅延パラメータの例 (s)
    dt = 0.01   # タイムステップの例 (s)
    j = 15      # 使用する基底関数の数
    J = 15      # 基底関数の総数
    tau = 1.0   # タイムウィンドウの長さ (s)

    kernel, t_axis = main(alphas, delta, dt, tau, j, J)

    plt.plot(t_axis, kernel[::-1])  # x軸に対応させるため反転戻す
    plt.title('FLNK Filter Kernel')
    plt.xlabel('Time (s)')
    plt.ylabel('Kernel Value')
    plt.grid()
    plt.show()