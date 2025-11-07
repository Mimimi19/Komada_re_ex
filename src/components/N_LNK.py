from matplotlib import pyplot as plt
import numpy as np
from scipy.special import erf

def main(x_input, a, kappa, b1, b2, ka):
    """
    非線形モデルの計算を行います。
    x_input: 入力値（スカラーまたはNumPy配列）
    a, kappa, b1, b2: 非線形パラメータ、
    a:シグモイド関数の指数部に含まれるパラメーター。この指数は $0$ から $20$ の間で変化します
    b1：入力に対するオフセット（閾値）を制御するパラメーター。
    kappa：入力のスケール（傾きや鋭さ）**を制御するパラメーター。
    b2: 関数の出力オフセットを制御するパラメーター
    ka: スケーリングパラメータ, 活性化密度関数
    x_inputがNumPy配列の場合、要素ごとに計算が適用されます（ベクトル化）。
    """
    erf_result = erf(kappa*x_input +b1) +1
    return (a**erf_result)*ka /2 + b2
if __name__ == "__main__":
    # --- 94-parents.txt の非線形パラメータ (Model 1) ---
    # a: x[J+1], b1: x[J+2], b2: x[J+3]
    # J=15を仮定 (config['J']から)
    J = 15
    params_94_parents = np.array([
        0.992053, 0.057222, 0.214281, 0.240299, 0.425180,
        0.385704, 0.017517, 0.696799, 0.247215, 0.175373,
        0.424967, 0.153786, 0.064951, 0.451844, 0.585853, # alphas
        0.180307, # delta
        4.209184, 0.665612, -0.281921, # a, b1, b2
        0.625779, 0.948792, 0.221268 # 動的パラメータ
    ])
    a_94, b1_94, b2_94 = params_94_parents[J+1], params_94_parents[J+2], params_94_parents[J+3]

    # --- 03-params.txt の非線形パラメータ (Model 2) ---
    params_03_params = np.array([
        0.850741, 0.043609, 0.314961, 0.115735, 0.405859,
        0.181998, 0.488759, 0.201215, 0.388303, 0.646640,
        0.021255, 0.884164, 0.723802, 0.443743, 0.024795, # alphas
        0.083580, # delta
        9.336384, 4.152542, -0.673780, # a, b1, b2
        0.419078, 0.164253, 0.170182 # 動的パラメータ
    ])
    a_03, b1_03, b2_03 = params_03_params[J+1], params_03_params[J+2], params_03_params[J+3]

    # 非線形変換の入力範囲を生成 (例: -5 から 5)
    x_input_range = np.linspace(-5, 5, 100)

    # --- グラフの生成 ---
    plt.figure(figsize=(10, 6))

    # 94-parents.txt の非線形変換をプロット
    output_94 = main(x_input_range, a_94, b1_94, b2_94)
    plt.plot(x_input_range, output_94, label='Nonlinear Function (94-parents.txt)', color='orange', linestyle='--')

    # 03-params.txt の非線形変換をプロット
    output_03 = main(x_input_range, a_03, b1_03, b2_03)
    plt.plot(x_input_range, output_03, label='Nonlinear Function (03-params.txt)', color='red', linestyle=':')

    plt.title('Comparison of Nonlinear Functions')
    plt.xlabel('Input (g)')
    plt.ylabel('Output (U_Nonlinear)')
    plt.grid()
    plt.legend()
    plt.show()