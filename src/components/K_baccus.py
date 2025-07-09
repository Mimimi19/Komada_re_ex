
import numpy as np
import tqdm

def dP(R, A, I1, I2, dt, u, ka, kfi, kfr, ksi, ksr):
    # 状態変数の更新
    dR_dt = (-ka * R * u + kfr * I1) * dt
    dA_dt = (ka * R * u - kfi * A) * dt
    dI1_dt = (kfi * A - kfr * I1 - ksi * I1) * dt
    dI2_dt = (ksi * I1 - ksr * I2) * dt

    return [dR_dt, dA_dt, dI1_dt, dI2_dt]

def main(time_steps, u_input, dt, R_start, A_start, I1_start, I2_start, ka, kfi, kfr, ksi, ksr):
    """
    Baccusモデルの状態微分方程式を定義します。
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
    """
    check = 1
    #Kineticモデル
    #初期値の入力    
    keep_R = R_start
    keep_A = A_start
    keep_I1 = I1_start
    keep_I2 = I2_start

    R_state = np.array([keep_R])
    A_state = np.array([keep_A])
    I1_state = np.array([keep_I1])
    I2_state = np.array([keep_I2])

    #4状態の計算
    for i in tqdm.tqdm(range(1, time_steps)):
        # ルンゲ・クッタ法
        # 1段階目
        Runge1_R, Runge1_A, Runge1_I1, Runge1_I2 = dP(keep_R, keep_A, keep_I1, keep_I2, dt, u_input[i], ka, kfi, kfr, ksi, ksr)
        # 2段階目
        Runge2_R, Runge2_A, Runge2_I1, Runge2_I2 = dP(keep_R + Runge1_R / 2, keep_A + Runge1_A / 2, keep_I1 + Runge1_I1 / 2, keep_I2 + Runge1_I2 / 2, dt, u_input[i], ka, kfi, kfr, ksi, ksr)
        # 3段階目
        Runge3_R, Runge3_A, Runge3_I1, Runge3_I2 = dP(keep_R + Runge2_R / 2, keep_A + Runge2_A / 2, keep_I1 + Runge2_I1 / 2, keep_I2 + Runge2_I2 / 2, dt, u_input[i], ka, kfi, kfr, ksi, ksr)
        # 4段階目
        Runge4_R, Runge4_A, Runge4_I1, Runge4_I2 = dP(keep_R + Runge3_R, keep_A + Runge3_A, keep_I1 + Runge3_I1, keep_I2 + Runge3_I2, dt, u_input[i], ka, kfi, kfr, ksi, ksr)
        
        # 状態の更新
        keep_R += (Runge1_R + 2 * Runge2_R + 2 * Runge3_R + Runge4_R) / 6
        keep_A += (Runge1_A + 2 * Runge2_A + 2 * Runge3_A + Runge4_A) / 6
        keep_I1 += (Runge1_I1 + 2 * Runge2_I1 + 2 * Runge3_I1 + Runge4_I1) / 6
        keep_I2 += (Runge1_I2 + 2 * Runge2_I2 + 2 * Runge3_I2 + Runge4_I2) / 6
        
        # 状態のクリッピング
        # 状態変数は0から1の範囲に制限する
        keep_R = np.clip(keep_R, 0.0, 1.0)
        keep_A = np.clip(keep_A, 0.0, 1.0)
        keep_I1 = np.clip(keep_I1, 0.0, 1.0)
        keep_I2 = np.clip(keep_I2, 0.0, 1.0)
        
        
        # 3状態に異常がないかのチェック
        if not (0 <= keep_R < 1 and 0 <= keep_A < 1 and 0 <= keep_I1 < 1 and 0 <= keep_I2 < 1):
            check = 0
            print(f"State out of bounds at step {i}: R={keep_R}, A={keep_A}, I1={keep_I1}, I2={keep_I2}")
            break
        # 状態の保存
        R_state = np.append(R_state, keep_R)
        A_state = np.append(A_state, keep_A)
        I1_state = np.append(I1_state, keep_I1)
        I2_state = np.append(I2_state, keep_I2)
    
    return R_state, A_state, I1_state, I2_state, check

if __name__ == "__main__":
    # テスト用のパラメータ
    time = 100
    u = np.random.rand(time)
    dt = 0.01
    R = 1.0
    A = 0.0
    I1 = 0.0
    I2 = 0.0
    ka = 0.5
    kfi = 0.1
    kfr = 0.05
    ksi = 0.02
    ksr = 0.01
    
    R_state, A_state, I1_state, I2_state, check = main(time, u, dt, R, A, I1, I2, ka, kfi, kfr, ksi, ksr)
    print(f"Check status for K_baccus: {check}")
    print(f"Last R state: {R_state[-1]}")
    print(f"Last A state: {A_state[-1]}")
    print(f"Last I1 state: {I1_state[-1]}")
    print(f"Last I2 state: {I2_state[-1]}")