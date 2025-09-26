# BasisFunctions.py
import numpy as np
import matplotlib.pyplot as plt

# 基底関数を定義するモジュール　式13
def main(t, j, tau):
    t = np.asarray(t)  # 入力をnumpy配列に変換
    f_x = np.zeros_like(t, dtype=float)  # 出力配列の初期化

    mask = (0 <= t) & (t <= tau)
    f_x[mask] = np.sin(np.pi * j * (2 * t[mask] / tau - (t[mask] / tau) ** 2))

    return f_x

if __name__ == "__main__":
    t = 0.5
    j = 15
    tau = 1.0
    result = main(t, j, tau)

    # グラフを描画する
    t_values = np.linspace(0, tau, 100)
    results = [main(t, j, tau) for t in t_values]

    plt.plot(t_values, results, label=f"j={j}, tau={tau}")
    plt.xlabel("Time (t)")
    plt.ylabel("Basis Function Value")
    plt.title("Basis Function")
    plt.legend()
    plt.grid()
    plt.show()
