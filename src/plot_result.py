import matplotlib
# サーバー環境(画面なし)でも動くようにバックエンドを指定
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle

# codeディレクトリへのパスを通す
sys.path.append('../code')
try:
    import cellclass
except ImportError:
    sys.path.append('./code')

def main():
    # コマンドライン引数でファイル名を指定できるように変更
    # 使い方: python check_result.py <ファイル名>
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # 指定がない場合のデフォルト
        filename = 'cb1_LNK_LNK_g2_100.pkl'
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    print(f"Loading {filename}...")
    with open(filename, 'rb') as f:
        cell = pickle.load(f)

    # 保存用ファイル名の作成 (例: data.pkl -> data.png)
    save_filename = os.path.splitext(filename)[0] + ".png"

    print("Processing results...")
    
    # --- データの取得 ---
    if hasattr(cell, 'mp') and cell.mp is not None:
        y_data = cell.mp.copy()
    elif hasattr(cell, 'fr') and cell.fr is not None:
        y_data = cell.fr.copy()
    else:
        print("Error: No response data found.")
        return

    if hasattr(cell, 'model_resp') and cell.model_resp is not None:
        y_model = cell.model_resp.copy()
    else:
        print("Warning: No model response found.")
        return

    # --- 正規化 (Z-score Normalization) ---
    if np.std(y_data) != 0:
        y_data_norm = (y_data - np.mean(y_data)) / np.std(y_data)
    else:
        y_data_norm = y_data - np.mean(y_data)

    if np.std(y_model) != 0:
        y_model_norm = (y_model - np.mean(y_model)) / np.std(y_model)
    else:
        y_model_norm = y_model - np.mean(y_model)
    # ------------------------------------

    dt = getattr(cell, 'dt', 0.001)
    time_axis = np.arange(len(y_data)) * dt

    # --- プロットと保存 ---
    plt.figure(figsize=(12, 8))

    # 1. 全体図
    plt.subplot(2, 1, 1)
    plot_len = min(int(30/dt), len(y_data))
    
    plt.plot(time_axis[:plot_len], y_data_norm[:plot_len], 'k-', alpha=0.6, label='Actual (Norm)')
    plt.plot(time_axis[:plot_len], y_model_norm[:plot_len], 'r--', lw=1.5, label='Model (Norm)')
    plt.title(f"Model Fitting: {filename}")
    plt.ylabel('Normalized Response')
    plt.legend()
    plt.grid(True)

    # 2. 拡大図
    plt.subplot(2, 1, 2)
    start_sec = 5
    end_sec = 10
    start_idx = int(start_sec/dt)
    end_idx = int(end_sec/dt)
    
    if end_idx < len(y_data):
        plt.plot(time_axis[start_idx:end_idx], y_data_norm[start_idx:end_idx], 'k-', alpha=0.6, label='Actual')
        plt.plot(time_axis[start_idx:end_idx], y_model_norm[start_idx:end_idx], 'r--', lw=2, label='Model')
        plt.xlabel("Time (s)")
        plt.ylabel('Normalized Response')
        plt.title(f"Zoomed ({start_sec}-{end_sec} sec)")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # 画像として保存
    plt.savefig(save_filename)
    print(f"Graph saved to: {save_filename}")

if __name__ == "__main__":
    main()