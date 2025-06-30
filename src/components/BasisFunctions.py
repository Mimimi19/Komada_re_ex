import numpy as np
import matplotlib.pyplot as plt

# 基底関数を定義するモジュール　式13
def main(t, j, tau=1.0):
    if 0<= t<= tau:
        return np.sin(np.pi * j * ( 2 * t / tau - ( t /tau )**2))
    else:
        return 0.0

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
