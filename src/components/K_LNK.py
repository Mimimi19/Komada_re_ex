

def main(P, t_val, u_t_val, ka, kfi, kfr, ksi, ksr):
    """
    動力学ブロックの常微分方程式系を定義します。
    P: 現在の状態占有率ベクトル [R, A, I1, I2]
    t_val: 現在の時刻
    u_t_val: u(t) を補間する関数
    ka, kfi, kfr, ksi, ksr: 動力学ブロックのレート定数
    """
    R, A, I1, I2 = P
    
    # 時刻 t_val での u(t) の値を補間により取得
    u_val = u_t_val(t_val)
    # レート定数は常に非負であるべき
    u_val = max(0, u_val) 

    # 状態の微分方程式（式11に基づき、明示的に記述）
    # ソースの行列Qは、P(t)Qの形式ですが、多くのODEソルバーはdP/dt = f(P, t) の形式を期待します。
    # そして、状態占有率の合計は常に1であるべきです。
    dR_dt = -u_val * ka * R + kfr * I1
    dA_dt = u_val * ka * R - kfi * A
    dI1_dt = kfi * A - kfr * I1 - ksi * I1 + u_val * ksr * I2
    dI2_dt = ksi * I1 - u_val * ksr * I2
    
    return [dR_dt, dA_dt, dI1_dt, dI2_dt]