import numpy as np
from scipy.signal import resample
import os
import sys

def main():
    # --- 1. ファイルパスの設定 ---
     # 入力ファイルパス
    stim_path = 'data/ret2p/chirp_stimulus.txt'     # 1000Hz
    resp_path = 'data/ret2p/response_data_repeat_1.txt' # 64Hz

    output_stim_path = 'data/ret2p/chirp_stim_64Hz.txt'
    output_resp_path = 'data/ret2p/response_data_64Hz.txt'
    
    # --- 2. ファイル存在チェック ---
    # ファイルがない場合は、エラーメッセージを出してプログラムを強制終了します
    if not os.path.exists(stim_path):
        print(f"【エラー】入力ファイルが見つかりません: {stim_path}")
        print(f"現在の作業ディレクトリ: {os.getcwd()}")
        sys.exit(1) # ここで終了

    if not os.path.exists(resp_path):
        print(f"【エラー】入力ファイルが見つかりません: {resp_path}")
        print(f"現在の作業ディレクトリ: {os.getcwd()}")
        sys.exit(1) # ここで終了

    # --- 3. データの読み込み ---
    print(f"読み込み中: {stim_path}")
    stim = np.loadtxt(stim_path)
    
    print(f"読み込み中: {resp_path}")
    resp = np.loadtxt(resp_path)
    
    # --- 4. データの整形 ---
    # 応答データが2次元(複数ROI)の場合、最初のROI(0列目)を抽出
    if resp.ndim > 1:
        print(f"Response data shape: {resp.shape}. Using ROI 0.")
        resp = resp[:, 0]
    
    # 入力刺激が2次元の場合、1次元に平坦化
    if stim.ndim > 1:
        stim = stim.flatten()
        
    n_stim = len(stim)
    n_resp = len(resp)
    
    print(f"Original Stimulus length: {n_stim} (1000Hz assumed)")
    print(f"Original Response length: {n_resp} (64Hz assumed)")
    
    # --- 5. ダウンサンプリング ---
    # 刺激データの点数を、応答データの点数(n_resp)に合わせる
    print("ダウンサンプリングを実行します...")
    stim_64Hz = resample(stim, n_resp)
    
    print(f"Downsampled Stimulus length: {len(stim_64Hz)}")
    
    # --- 6. 保存 ---
    os.makedirs(os.path.dirname(output_stim_path), exist_ok=True)
    
    np.savetxt(output_stim_path, stim_64Hz, fmt='%.6f')
    np.savetxt(output_resp_path, resp, fmt='%.6f')
    
    print(f"Saved stimulus to {output_stim_path}")
    print(f"Saved response to {output_resp_path}")

    # 推奨dtの計算
    dt_calc = 32.0 / n_resp if n_resp > 0 else 0
    print(f"\n[設定値] ret2p-1.yamlの dt には {dt_calc:.6f} (約0.015625) を設定してください。")

if __name__ == "__main__":
    main()