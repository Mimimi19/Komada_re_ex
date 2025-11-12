# BasisFunctions.py
import numpy as np
import matplotlib.pyplot as plt

def f_x(t, j, tau):
    f_x = np.zeros_like(t, dtype=float)# 出力配列の初期化
    t_mask = (0 <= t) & (t <= tau) # 条件を満たすインデックスを取得 1 or 0
    f_x[t_mask] = np.sin(np.pi * j * (2 * t[t_mask] / tau - (t[t_mask] / tau) ** 2))
    return f_x

# 基底関数を定義するモジュール　
def main(t, J, tau):
    J = J + 1 # j=1からJまで計算するため
    f_x_list = [f_x(t, j, tau).reshape((1, -1)) for j in range(1, J)]
    return np.concatenate(f_x_list, axis=0)

if __name__ == "__main__":
    
    # 実行パラメータの設定
    J_max = 15      # j = 1 から J_max-1 までの基底関数をプロット
    tau = 15.0    # 期間の終点
    t_values = np.linspace(0, 20, 100)  # 横軸 t の値
    
    # J_max-1 個の基底関数の値を行列として計算 (各行が各 j の基底関数)
    results_matrix = main(t_values, J_max, tau)

    # グラフ描画
    plt.figure(figsize=(10, 6))
    
    # results_matrix の各行（つまり、各 j の基底関数）を取り出してプロット
    # j は 1 から J_max-1 まで動く
    for i in range(J_max ):
        j_val = i + 1  # 実際の j の値
        
        # results_matrix[i, :] は i番目の行 (j=i+1 の基底関数)
        plt.plot(t_values, results_matrix[i, :], label=f"$j={j_val}$")

    plt.xlabel("時間 ($t$)")
    plt.ylabel("基底関数の値 ($f_x$)")
    plt.title(f"複数の基底関数 ($j=1$ から $j={J_max-1}$ まで)")
    plt.legend(loc='lower left', ncol=2) # 凡例を表示
    plt.grid(True)
    plt.show()

