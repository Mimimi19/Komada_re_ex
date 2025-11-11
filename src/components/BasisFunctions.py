# BasisFunctions.py
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def f_x(t, j, tau):
    f_x = np.zeros_like(t, dtype=np.float64) # 出力配列の初期化
    
    t_mask = (0 <= t) & (t <= tau) # 条件を満たすインデックスを取得 1 or 0
    f_x[t_mask] = np.sin(np.pi * j * (2 * t[t_mask] / tau - (t[t_mask] / tau) ** 2))
    return f_x

# 基底関数を定義するモジュール　
@jit(nopython=True)
def main(t, J, tau):
    t_len = t.shape[0]
    results_matrix = np.empty((J, t_len), dtype=np.float64)
    
    #  Numbaが得意な for ループで配列を埋める
    for i in range(J):
        j_val = i + 1
        
        results_matrix[i, :] = f_x(t, j_val, tau)
    
    # 4. 確保した配列を返す
    return results_matrix

if __name__ == "__main__":
    
    # 実行パラメータの設定
    J_max = 15      # j = 1 から J_max までの基底関数をプロット
    tau = 15.0    # 期間の終点
    t_values = np.linspace(0, 20, 100)  # 横軸 t の値
    
    # J_max 個の基底関数の値を行列として計算 (各行が各 j の基底関数)
    results_matrix = main(t_values, J_max, tau)

    # グラフ描画
    plt.figure(figsize=(10, 6))
    
    # results_matrix の各行（つまり、各 j の基底関数）を取り出してプロット
    # j は 1 から J_max (15) まで動く
    for i in range(J_max):
        j_val = i + 1  # 実際の j の値
        
        # results_matrix[i, :] は i番目の行 (j=i+1 の基底関数)
        plt.plot(t_values, results_matrix[i, :], label=f"$j={j_val}$")

    plt.xlabel("時間 ($t$)")
    plt.ylabel("基底関数の値 ($f_x$)")
    plt.title(f"複数の基底関数 ($j=1$ から $j={J_max}$ まで)") 
    plt.legend(loc='lower left', ncol=2) # 凡例を表示
    plt.grid(True)
    plt.show()