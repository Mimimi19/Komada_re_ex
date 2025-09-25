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
    # Baccus.yamlから読み込むパラメータの例
    dt = 0.0002 # タイムステップ (s)
    tau = 1.0   # タイムウィンドウの長さ (s)
    J = 15      # 基底関数の総数 (config['J']から)

    # --- 94-parents.txt のパラメータ (Model 1) ---
    # alphas: x[0]からx[J-1]
    # delta: x[J]
    params_94_parents = np.array([
        0.992053, 0.057222, 0.214281, 0.240299, 0.425180,
        0.385704, 0.017517, 0.696799, 0.247215, 0.175373,
        0.424967, 0.153786, 0.064951, 0.451844, 0.585853, # alphas (J=15個)
        0.180307, # delta
        4.209184, 0.665612, -0.281921, # 非線形パラメータ (a, b1, b2)
        0.625779, 0.948792, 0.221268 # 動的パラメータ (ka, kfi, kfr)
    ])
    alphas_94 = params_94_parents[0:J]
    delta_94 = params_94_parents[J]

    # --- 03-params.txt のパラメータ (Model 2) ---
    params_03_params = np.array([
        0.850741, 0.043609, 0.314961, 0.115735, 0.405859,
        0.181998, 0.488759, 0.201215, 0.388303, 0.646640,
        0.021255, 0.884164, 0.723802, 0.443743, 0.024795, # alphas (J=15個)
        0.083580, # delta
        9.336384, 4.152542, -0.673780, # 非線形パラメータ (a, b1, b2)
        0.419078, 0.164253, 0.170182 # 動的パラメータ (ka, kfi, kfr)
    ])
    alphas_03 = params_03_params[0:J]
    delta_03 = params_03_params[J]

    # --- グラフの生成 ---
    plt.figure(figsize=(12, 6))

    # 94-parents.txt のフィルタカーネルをプロット
    kernel_94, t_axis_94 = main(alphas_94, delta_94, dt, tau, J, J)
    plt.plot(t_axis_94, kernel_94[::-1], label='FLNK Filter Kernel (94-parents.txt)', color='orange', linestyle='--')

    # 03-params.txt のフィルタカーネルをプロット
    kernel_03, t_axis_03 = main(alphas_03, delta_03, dt, tau, J, J)
    plt.plot(t_axis_03, kernel_03[::-1], label='FLNK Filter Kernel (03-params.txt)', color='red', linestyle=':')

    plt.title('Comparison of FLNK Filter Kernels')
    plt.xlabel('Time (s)')
    plt.ylabel('Kernel Value')
    plt.grid()
    plt.legend()
    plt.show()