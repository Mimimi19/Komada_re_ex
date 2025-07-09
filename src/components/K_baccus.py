import numpy as np
import tqdm
import sys # sysモジュールをインポート (tqdm.write() のために必要)

def dP(R, A, I1, I2, dt, u, ka, kfi, kfr, ksi, ksr):
    # 状態変数の更新を計算
    dR_dt = (-ka * R * u + kfr * I1) * dt
    dA_dt = (ka * R * u - kfi * A) * dt
    dI1_dt = (kfi * A - kfr * I1 - ksi * I1) * dt
    dI2_dt = (ksi * I1 - ksr * I2) * dt

    return [dR_dt, dA_dt, dI1_dt, dI2_dt]

def main(time_steps, u_input, dt, R_start, A_start, I1_start, I2_start, ka, kfi, kfr, ksi, ksr, label):
    """
    Baccusモデルの状態微分方程式を数値積分します。
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
    check = 1 # 計算が正常に完了したかどうかのフラグ
    # Kineticモデルの初期値設定
    keep_R = R_start
    keep_A = A_start
    keep_I1 = I1_start
    keep_I2 = I2_start

    # 状態を保存するためのNumPy配列を事前に確保
    # time_stepsのサイズで初期化し、ループ内でインデックスを使って値を設定します。
    # これにより、np.append()による頻繁なメモリ再割り当てとコピーを避けて高速化します。
    R_state = np.zeros(time_steps)
    A_state = np.zeros(time_steps)
    I1_state = np.zeros(time_steps)
    I2_state = np.zeros(time_steps)

    # 初期値を配列の最初の要素に設定
    R_state[0] = keep_R
    A_state[0] = keep_A
    I1_state[0] = keep_I1
    I2_state[0] = keep_I2

    # 4状態の計算ループ
    for i in tqdm.tqdm(range(1, time_steps), leave=False, desc=f'K_Model({label})'):
        # ルンゲ・クッタ法による次のタイムステップの状態変数の変化量を計算
        # 1段階目
        Runge1_R, Runge1_A, Runge1_I1, Runge1_I2 = dP(keep_R, keep_A, keep_I1, keep_I2, dt, u_input[i], ka, kfi, kfr, ksi, ksr)
        # 2段階目
        Runge2_R, Runge2_A, Runge2_I1, Runge2_I2 = dP(keep_R + Runge1_R / 2, keep_A + Runge1_A / 2, keep_I1 + Runge1_I1 / 2, keep_I2 + Runge1_I2 / 2, dt, u_input[i], ka, kfi, kfr, ksi, ksr)
        # 3段階目
        Runge3_R, Runge3_A, Runge3_I1, Runge3_I2 = dP(keep_R + Runge2_R / 2, keep_A + Runge2_A / 2, keep_I1 + Runge2_I1 / 2, keep_I2 + Runge2_I2 / 2, dt, u_input[i], ka, kfi, kfr, ksi, ksr)
        # 4段階目
        Runge4_R, Runge4_A, Runge4_I1, Runge4_I2 = dP(keep_R + Runge3_R, keep_A + Runge3_A, keep_I1 + Runge3_I1, keep_I2 + Runge3_I2, dt, u_input[i], ka, kfi, kfr, ksi, ksr)
        
        # 状態変数を更新
        keep_R += (Runge1_R + 2 * Runge2_R + 2 * Runge3_R + Runge4_R) / 6
        keep_A += (Runge1_A + 2 * Runge2_A + 2 * Runge3_A + Runge4_A) / 6
        keep_I1 += (Runge1_I1 + 2 * Runge2_I1 + 2 * Runge3_I1 + Runge4_I1) / 6
        keep_I2 += (Runge1_I2 + 2 * Runge2_I2 + 2 * Runge3_I2 + Runge4_I2) / 6
        
        # 状態のクリッピング: 状態変数を0から1の範囲に制限する
        keep_R = np.clip(keep_R, 0.0, 1.0)
        keep_A = np.clip(keep_A, 0.0, 1.0)
        keep_I1 = np.clip(keep_I1, 0.0, 1.0)
        keep_I2 = np.clip(keep_I2, 0.0, 1.0)
        
        # 状態が0から1の範囲を逸脱していないかのチェック (境界値を含むように <= に修正)
        # 浮動小数点数の誤差を考慮し、微小な範囲外を許容する場合もありますが、ここでは厳密にチェック
        if not (0 <= keep_R < 1 and 0 <= keep_A < 1 and 0 <= keep_I1 < 1 and 0 <= keep_I2 < 1):
            check = 0 # 異常フラグを設定
            # エラー発生時の状態を出力してデバッグしやすくする
            # tqdm.write() を使うことで、プログレスバーを乱さずにメッセージを出力
            tqdm.write(f"K_baccus: State out of bounds at step {i} (Label: {label}): "
                       f"R={keep_R:.4f}, A={keep_A:.4f}, I1={keep_I1:.4f}, I2={keep_I2:.4f}")
            # 計算が中断されたことを示すため、既に計算された部分の配列をスライスして返す
            return R_state[:i+1], A_state[:i+1], I1_state[:i+1], I2_state[:i+1], check
            
        # 状態の保存: 事前に確保した配列の現在のインデックスに直接書き込む
        R_state[i] = keep_R
        A_state[i] = keep_A
        I1_state[i] = keep_I1
        I2_state[i] = keep_I2
    
    return R_state, A_state, I1_state, I2_state, check

if __name__ == "__main__":
    # テスト用のパラメータ
    time_steps = 80000 
    u = np.random.rand(time_steps)
    dt = 0.0002
    R = 1.0
    A = 0.0
    I1 = 0.0
    I2 = 0.0
    ka = 0.5
    kfi = 0.1
    kfr = 0.05
    ksi = 0.02
    ksr = 0.01
    
    # main関数を呼び出してシミュレーションを実行
    R_state, A_state, I1_state, I2_state, check = main(time_steps, u, dt, R, A, I1, I2, ka, kfi, kfr, ksi, ksr, label="test_run")
    print(f"Check status for K_baccus: {check}")
    if check == 1: # 計算が正常に完了した場合のみ最終状態を表示
        print(f"Last R state: {R_state[-1]}")
        print(f"Last A state: {A_state[-1]}")
        print(f"Last I1 state: {I1_state[-1]}")
        print(f"Last I2 state: {I2_state[-1]}")
    else: # 計算が途中で失敗した場合
        print("Kinetic model simulation failed early.")
        print(f"Partial R state length: {len(R_state)}") 