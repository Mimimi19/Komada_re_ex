import numpy as np

def dP(u, dt, R, A, I1, I2, ka, kfi, kfr, ksi, ksr):
    # 状態変数の更新
    dR_dt = (-ka * R * u + kfr * I1)*dt
    dA_dt = (ka * R * u - kfi * A)*dt
    dI1_dt = (kfi * A - kfr * I1 - ksi * I1)*dt
    dI2_dt = (ksi * I1 - ksr * I2)*dt

    return [dR_dt, dA_dt, dI1_dt, dI2_dt]

def main(time, u, dt, R, A, I1, I2, ka, kfi, kfr, ksi, ksr):
    """
    Baccusモデルの状態微分方程式を定義します。
    R:休息状態の占有率
    A:活性状態の占有率
    I1:不活性化状態1の占有率
    I2:不活性化状態2の占有率
    dt: タイムステップ
    u: 入力信号（u(t)）
    ka: 活性化速度
    kfi: 高速不活性化速度
    kfr: 高速回復速度
    ksi: 低速活性化速度
    ksr: 超回復速度
    """
    #チェック用
    check = 1
    #Kineticモデル
    #初期値の入力
    keep_R = R
    keep_A = A
    keep_I1 = I1
    keep_I2 = I2

    #4状態の計算
    for i in range(time):
        if check == 1:
            
            if i == 0:
                R_state = np.append(R_state, keep_R)
                A_state = np.append(A_state, keep_A)
                I1_state = np.append(I1_state, keep_I1)
                I2_state = np.append(I2_state, keep_I2)

            #ルンゲ・クッタ法
            else:
                #1段階目
                Runge1_R, Runge1_A, Runge1_I1, Runge1_I2 = dP(keep_R, keep_A, keep_I1, keep_I2, dt, u[i], ka, kfi, kfr, ksi, ksr)
                #2段階目
                Runge2_R, Runge2_A, Runge2_I1, Runge2_I2 = dP(keep_R + Runge1_R / 2, keep_A + Runge1_A / 2, keep_I1 + Runge1_I1 / 2, keep_I2 + Runge1_I2 / 2, dt, u[i], ka, kfi, kfr, ksi, ksr)
                #3段階目
                Runge3_R, Runge3_A, Runge3_I1, Runge3_I2 = dP(keep_R + Runge2_R / 2, keep_A + Runge2_A / 2, keep_I1 + Runge2_I1 / 2, keep_I2 + Runge2_I2 / 2, dt, u[i], ka, kfi, kfr, ksi, ksr)
                #4段階目
                Runge4_R, Runge4_A, Runge4_I1, Runge4_I2 = dP(keep_R + Runge3_R, keep_A + Runge3_A, keep_I1 + Runge3_I1, keep_I2 + Runge3_I2, dt, u[i], ka, kfi, kfr, ksi, ksr)
                
                #状態の更新
                keep_R += (Runge1_R + 2 * Runge2_R + 2 * Runge3_R + Runge4_R) / 6
                keep_A += (Runge1_A + 2 * Runge2_A + 2 * Runge3_A + Runge4_A) / 6
                keep_I1 += (Runge1_I1 + 2 * Runge2_I1 + 2 * Runge3_I1 + Runge4_I1) / 6
                keep_I2 += (Runge1_I2 + 2 * Runge2_I2 + 2 * Runge3_I2 + Runge4_I2) / 6
            
            #3状態に異常がないかのチェック
            if keep_R < 0 or 1 < keep_R:
                check = 0
                break
                
            if keep_A < 0 or 1 < keep_A:
                check = 0
                break
            if keep_I1 < 0 or 1 < keep_I1:
                check = 0
                break
            if keep_I2 < 0 or 1 < keep_I2:
                check = 0
                break
            #状態の保存
            R_state = np.append(R_state, keep_R)
            A_state = np.append(A_state, keep_A)
            I1_state = np.append(I1_state, keep_I1)
            I2_state = np.append(I2_state, keep_I2)
    
    return R_state, A_state, I1_state, I2_state, check