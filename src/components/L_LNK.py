# L_LNK.py
import numpy as np
import matplotlib.pyplot as plt

import components.BasisFunctions as BasisFunctions

def QR_orthogonalization(vectors):
    """
    QR分解を用いて、高速かつ安定的に正規直交基底を求めます。
    
    np.linalg.qr は「列ベクトル」を直交化するため、
    入力 'vectors' (J, N) の「行ベクトル」を直交化するために
    一度転置(T)してからQR分解を実行し、結果を再び転置して返します。

    Args:
        vectors (np.ndarray): (J, N) の形状の配列。
                              J個の行ベクトルを正規直交化します。

    Returns:
        np.ndarray: (J, N) の形状の正規直交化されたベクトル（行ベクトル）。
    """
    
    # 1. (J, N) -> (N, J) へ転置
    #    QR分解が列ベクトル (J個) を処理できるようにするため。
    A = vectors.T
    
    # 2. QR分解を実行 (A = QR)
    #    Q は (N, J) の形状となり、その「列」が正規直交基底です。
    #    mode='reduced' は効率的な計算のために指定します。
    try:
        Q, R = np.linalg.qr(A, mode='reduced')
    except np.linalg.LinAlgError:
        # 入力ベクトルが線形従属などでQR分解に失敗した場合のフォールバック
        print("Warning: QR decomposition failed. Returning original vectors.")
        return vectors
        
    # 3. (N, J) -> (J, N) へ転置
    #    元の形状に戻し、「行」が正規直交基底となるようにします。
    orthogonal_vectors = Q.T
    
    return orthogonal_vectors

def main(alphas, delta, t, dt, tau):
    """
    FLNKフィルターカーネルを生成します（式12）
    alphas: 各基底関数の重み係数リスト (J個)
    delta: 細胞固有の遅延パラメータ (s)
    dt: タイムステップ (s)
    tau: フィルターの時間ウィンドウ (s)
    
    戻り値:
    - kernel[::-1]: 畳み込み用に反転されたカーネル
    - t_values[::-1]: 反転された時間軸
    """
    print(f"Linear Model -                 -                is being processed.", end='\r', flush=True)
    # alphasの長さが基底関数の総数 J となります
    J = len(alphas) 
    
    # 時間軸の生成: dataの長さtを dt刻みで分割
    t_values = np.arange(0, t*dt, dt)

    # 1. 時間軸の遅延
    shifted_t = t_values + delta
    
    # 2. 全ての基底関数の計算 (j=1 から J まで)
    # f_x_matrix の shape は (J-1) x len(shifted_t)
    f_x_matrix = BasisFunctions.main(shifted_t, J , tau)
    f_x_matrix = QR_orthogonalization(f_x_matrix)
    # 4. カーネルの計算 F_LNK (t) = Σ α_j f_x(t, j)
    kernel = np.dot(alphas, f_x_matrix)

    # 畳み込み用にフィルターを反転して返す
    return kernel[::-1], t_values[::-1]

# 適当に生成した動作確認用テストデータ
if __name__ == "__main__":
    # NumPy の　インポートを確認
    if 'np' not in globals():
        import numpy as np
    if 'plt' not in globals():
        import matplotlib.pyplot as plt

    # Baccus.yamlから読み込むパラメータの例
    t = 5000   # データの長さ (点数)
    dt = 0.0002 # タイムステップ (s)
    tau = 1.0   # タイムウィンドウの長さ (s)
    J = 15      # 基底関数の総数 (config['J']から)
    
    # --- 94-parents.txt のパラメータ (Model 1) ---
    params_94_parents = np.array([
        # ... (alphas: J=15個)
        0.992053, 0.057222, 0.214281, 0.240299, 0.425180,
        0.385704, 0.017517, 0.696799, 0.247215, 0.175373,
        0.424967, 0.153786, 0.064951, 0.451844, 0.585853,
        0.180307, # delta
        # ... (その他)
    ])
    alphas_94 = params_94_parents[0:J]
    delta_94 = params_94_parents[J]

    # --- 03-params.txt のパラメータ (Model 2) ---
    params_03_params = np.array([
        # ... (alphas: J=15個)
        0.850741, 0.043609, 0.314961, 0.115735, 0.405859,
        0.181998, 0.488759, 0.201215, 0.388303, 0.646640,
        0.021255, 0.884164, 0.723802, 0.443743, 0.024795,
        0.083580, # delta
        # ... (その他)
    ])
    alphas_03 = params_03_params[0:J]
    delta_03 = params_03_params[J]

    # --- グラフの生成 ---
    plt.figure(figsize=(10, 6))

    # Model 1 のフィルタカーネルをプロット
    # main関数に kernel_points を追加
    kernel_94, t_axis_94 = main(alphas_94, delta_94, t, dt, tau)
    # t_axis_94 は反転されているため、カーネルも反転前の値 kernel_94 をプロット
    plt.plot(t_axis_94, kernel_94, label='FLNK Filter Kernel (Model 1: 94-parents.txt)', color='orange')

    # Model 2 のフィルタカーネルをプロット
    kernel_03, t_axis_03 = main(alphas_03, delta_03, t, dt, tau)
    plt.plot(t_axis_03, kernel_03, label='FLNK Filter Kernel (Model 2: 03-params.txt)', color='blue', linestyle='--')

    plt.title('Comparison of FLNK Filter Kernels')
    # 時間軸は逆転している (tau から 0 へ向かう)
    plt.xlabel('Time (s) from Stimulus') 
    plt.ylabel('Kernel Value')
    plt.grid(True)
    plt.legend()
    plt.show()