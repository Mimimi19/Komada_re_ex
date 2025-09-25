import numpy as np # math.exp の代わりに np.exp を使用

def main(x_input, a, b1, b2):
    """
    非線形モデルの計算を行います。
    x_input: 入力値（スカラーまたはNumPy配列）
    a, b1, b2: 非線形パラメータ

    x_inputがNumPy配列の場合、要素ごとに計算が適用されます（ベクトル化）。
    """
    # np.exp はスカラーとNumPy配列の両方に対応し、要素ごとの計算を行う
    return a / (1 + b1 * np.exp(-b2 * x_input))

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