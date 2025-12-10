import numpy as np
import os
import sys

def main():
    # --- 1. ファイルパスの設定 ---
    stim_path = 'data/ret2p/chirp_stimulus.txt'          # 入力 (1000Hz)
    resp_path = 'data/ret2p/response_data_repeat_1.txt' # 目標サイズ (64Hz)

    output_stim_path = 'data/ret2p/chirp_stim_64Hz_linear.txt' # 出力ファイル名
    output_resp_path = 'data/ret2p/response_data_64Hz.txt'
    
    # --- 2. ファイル存在チェック ---
    if not os.path.exists(stim_path):
        print(f"【エラー】入力ファイルが見つかりません: {stim_path}")
        sys.exit(1)

    if not os.path.exists(resp_path):
        print(f"【エラー】入力ファイルが見つかりません: {resp_path}")
        sys.exit(1)

    # --- 3. データの読み込み ---
    print(f"読み込み中: {stim_path}")
    stim = np.loadtxt(stim_path)
    
    print(f"読み込み中: {resp_path}")
    resp = np.loadtxt(resp_path)
    
    # --- 4. データの整形 ---
    if resp.ndim > 1:
        resp = resp[:, 0] # 最初のROIのみ使用
    
    if stim.ndim > 1:
        stim = stim.flatten()
        
    n_stim = len(stim)
    n_resp = len(resp)
    
    print(f"Original Stimulus length: {n_stim}")
    print(f"Target Response length: {n_resp}")
    
    # --- 5. 線形補間によるダウンサンプリング ---
    print("線形補間(Linear Interpolation)を実行します...")
    
    # 元のデータの座標 (0 から 1 まで等間隔)
    x_old = np.linspace(0, 1, n_stim)
    
    # 新しいデータの座標 (0 から 1 まで、n_resp個の点で等間隔)
    x_new = np.linspace(0, 1, n_resp)
    
    # 線形補間を実行
    # x_new の各点において、x_old と stim の関係に基づき値を計算
    stim_64Hz = np.interp(x_new, x_old, stim)
    
    print(f"Downsampled Stimulus length: {len(stim_64Hz)}")
    
    # --- 6. 保存 ---
    os.makedirs(os.path.dirname(output_stim_path), exist_ok=True)
    
    np.savetxt(output_stim_path, stim_64Hz, fmt='%.6f')
    np.savetxt(output_resp_path, resp, fmt='%.6f')
    
    print(f"Saved stimulus to {output_stim_path}")
    print(f"Saved response to {output_resp_path}")

    # dtの計算
    dt_calc = 32.0 / n_resp if n_resp > 0 else 0
    print(f"\n[設定値] ret2p-1.yamlの dt には {dt_calc:.6f} を設定してください。")
    print(f"input_file も '{output_stim_path}' に変更してください。")

if __name__ == "__main__":
    main()