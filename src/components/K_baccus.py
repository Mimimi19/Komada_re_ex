# components/K_baccus.py
import numpy as np
import time # timeモジュールを追加
from numba import jit # Numbaのjitデコレータをインポート

# 4変数 (R, A, I1, I2) の微分方程式
# 先行研究寄りに、A_state の時間変化をそのまま出力として扱えるようにする
@jit(nopython=True, cache=True)
def dP(R, A, I1, I2, dt, u, ka, kfi, kfr, ksi, ksr):
    # 状態変数の更新を計算
    dR_dt = (-ka * R * u + kfr * I1) * dt
    dA_dt = (ka * R * u - kfi * A) * dt
    dI1_dt = (kfi * A + ksr * I2 * u - kfr * I1 - ksi * I1) * dt
    dI2_dt = (ksi * I1 - ksr * I2 * u) * dt

    return dR_dt, dA_dt, dI1_dt, dI2_dt

@jit(nopython=True, cache=True)
def _simulation_loop_jit(time_steps, u_input, dt,
                         R_start, A_start, I1_start, I2_start,
                         ka, kfi, kfr, ksi, ksr):
    """Numbaで最適化されたコア計算ループ（RK4）"""
    check = 1 # 計算が正常に完了したかどうかのフラグ
    keep_R, keep_A, keep_I1, keep_I2 = R_start, A_start, I1_start, I2_start

    R_state = np.zeros(time_steps)
    A_state = np.zeros(time_steps)
    I1_state = np.zeros(time_steps)
    I2_state = np.zeros(time_steps)

    R_state[0], A_state[0], I1_state[0], I2_state[0] = keep_R, keep_A, keep_I1, keep_I2

    # 4状態の計算ループ (tqdmを削除)
    last_idx = 0
    for i in range(1, time_steps):
        # --- RK4 ---
        k1_R, k1_A, k1_I1, k1_I2 = dP(keep_R, keep_A, keep_I1, keep_I2, dt, u_input[i], ka, kfi, kfr, ksi, ksr)
        k2_R, k2_A, k2_I1, k2_I2 = dP(keep_R + k1_R / 2.0, keep_A + k1_A / 2.0, keep_I1 + k1_I1 / 2.0, keep_I2 + k1_I2 / 2.0,
                                      dt, u_input[i], ka, kfi, kfr, ksi, ksr)
        k3_R, k3_A, k3_I1, k3_I2 = dP(keep_R + k2_R / 2.0, keep_A + k2_A / 2.0, keep_I1 + k2_I1 / 2.0, keep_I2 + k2_I2 / 2.0,
                                      dt, u_input[i], ka, kfi, kfr, ksi, ksr)
        k4_R, k4_A, k4_I1, k4_I2 = dP(keep_R + k3_R, keep_A + k3_A, keep_I1 + k3_I1, keep_I2 + k3_I2,
                                      dt, u_input[i], ka, kfi, kfr, ksi, ksr)

        keep_R += (k1_R + 2.0 * k2_R + 2.0 * k3_R + k4_R) / 6.0
        keep_A += (k1_A + 2.0 * k2_A + 2.0 * k3_A + k4_A) / 6.0
        keep_I1 += (k1_I1 + 2.0 * k2_I1 + 2.0 * k3_I1 + k4_I1) / 6.0
        keep_I2 += (k1_I2 + 2.0 * k2_I2 + 2.0 * k3_I2 + k4_I2) / 6.0

        # 占有率が0-1の範囲外になった場合、計算を打ち切る
        # （厳密に 1.0 を含めるかは元コードに合わせ、ここでは 0 <= x < 1 とする）
        if not (0.0 <= keep_R < 1.0 and 0.0 <= keep_A < 1.0 and 0.0 <= keep_I1 < 1.0 and 0.0 <= keep_I2 < 1.0):
            check = 0 # 異常フラグ
            last_idx = i
            break

        R_state[i], A_state[i], I1_state[i], I2_state[i] = keep_R, keep_A, keep_I1, keep_I2
        last_idx = i

    return R_state, A_state, I1_state, I2_state, check, last_idx

def main(time_steps, u_input, dt,
         R_start, A_start, I1_start, I2_start,
         ka, kfi, kfr, ksi, ksr,
         label):
    """
    Baccusモデルの状態微分方程式を数値積分します。
    この関数は、tqdmで進捗を表示し、dP関数はNumbaで最適化されています。
    R: 休息状態の占有率
    A: 活性状態の占有率
    I1: 不活性化状態1の占有率
    I2: 不活性化状態2の占有率
    dt: タイムステップ
    u_input: 入力信号（u(t)）
    ka: 活性化速度
    kfi: 高速不活性化速度
    kfr: 高速回復速度
    ksi: 低速活性化速度
    ksr: 超回復速度
    label: tqdmのdescに表示する追加ラベル
    """
    # flush=Trueで、バッファリングされずにすぐ表示されるようにする
    # end='\r'で描き終わったらカーソルを行頭に戻す
    print(f'\r\033[Running: K_Model({label})', end='\r', flush=True)

    # Numbaで最適化されたコア計算ループを呼び出す
    R_state, A_state, I1_state, I2_state, check, last_idx = _simulation_loop_jit(
        time_steps, u_input, dt,
        R_start, A_start, I1_start, I2_start,
        ka, kfi, kfr, ksi, ksr
    )


    # 計算が途中で失敗した場合、結果を正しい長さでスライスする
    if check == 0:
        return (R_state[:last_idx+1],
                A_state[:last_idx+1],
                I1_state[:last_idx+1],
                I2_state[:last_idx+1],
                check)

    return R_state, A_state, I1_state, I2_state, check


if __name__ == "__main__":
    # テスト用のパラメータ
    time_steps = 80000
    u = np.random.rand(time_steps)
    dt = 0.0002
    R, A, I1, I2 = 1.0, 0.0, 0.0, 0.0
    ka, kfi, kfr, ksi, ksr = 0.5, 0.1, 0.05, 0.02, 0.01

    print("シミュレーションを開始します...")
    start_time = time.time()

    # main関数を呼び出してシミュレーションを実行
    R_state, A_state, I1_state, I2_state, check = main(
        time_steps, u, dt, R, A, I1, I2,
        ka, kfi, kfr, ksi, ksr,
        label="test_run"
    )

    end_time = time.time()
    print(f"シミュレーションが完了しました。 (実行時間: {end_time - start_time:.4f}秒)")

    print(f"Check status for K_baccus: {check}")
    if check == 1:
        print(f"Last R state: {R_state[-1]}")
        print(f"Last A state: {A_state[-1]}")
        print(f"Last I1 state: {I1_state[-1]}")
        print(f"Last I2 state: {I2_state[-1]}")
    else:
        print("Kinetic model simulation failed early.")
        print(f"Partial R state length: {len(R_state)}")
        print("状態が範囲外になったため、計算が中断されました。")
