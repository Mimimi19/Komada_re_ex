import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import japanize_matplotlib

 # Macの場合の日本語フォント設定例
"""
uv run data/ret2p/plot.py \
    --stim data/ret2p/trim/chirp_stim_64Hz_bilinear.txt \
    --resp data/ret2p/trim/response_data_64Hz.txt   
"""

# --- 1. 日本語フォントの設定 ---
# Windowsなら'MS Gothic', Macなら'Hiragino Sans'などを指定します
# 環境に合わせて適宜変更してください
plt.rcParams['font.family'] = 'Hiragino Sans' 
# --- 2. ファイルとパラメータの設定 ---
stim_filename = 'data/ret2p/trim/chirp_stim_64Hz_bilinear.txt'
resp_filename = 'data/ret2p/trim/response_data_64Hz.txt'

# サンプリングレート (Hz)
# ※データに合わせて変更してください。
# ここでは提示されたコード(上部)に従い、両方とも64Hzにダウンサンプリング済みと仮定します。
stim_sampling_rate = 64  
resp_sampling_rate = 64

# プロット対象のROIインデックス (0が最初の細胞)
target_roi_index = 1

# --- 3. データの読み込み ---
try:
    # 刺激データの読み込み
    chirp_stim = np.loadtxt(stim_filename)
    stim_time = np.arange(0, len(chirp_stim)) / stim_sampling_rate
    
    # 応答データの読み込み
    response_data = np.loadtxt(resp_filename)

    # 応答データが1次元（1列のみ）か2次元（複数列）かで処理を分ける
    if response_data.ndim == 1:
        selected_roi_response = response_data
        roi_label = "ROI 0 (抽出データ)"
    else:
        # 指定したインデックスが存在するか確認
        if target_roi_index < response_data.shape[1]:
            selected_roi_response = response_data[:, target_roi_index]
            roi_label = f"ROI {target_roi_index}"
        else:
            print(f"警告: ROI {target_roi_index} は存在しません。最初のROIを表示します。")
            selected_roi_response = response_data[:, 0]
            roi_label = "ROI 0"

    resp_time = np.arange(0, len(selected_roi_response)) / resp_sampling_rate

except FileNotFoundError as e:
    print(f"エラー: ファイルが見つかりません。\n詳細: {e}")
    # 以降の処理を中断するためにexitするか、ダミーデータを入れる等の対応が必要
    # ここでは空の配列を入れてエラー落ちを防ぎます
    chirp_stim = np.array([])
    selected_roi_response = np.array([])
    stim_time = np.array([])
    resp_time = np.array([])

# --- 4. グラフのプロット (上下に並べる) ---
if len(chirp_stim) > 0 and len(selected_roi_response) > 0:
    plt.figure(figsize=(12, 8))

    # 上段: 刺激波形
    plt.subplot(2, 1, 1)
    plt.plot(stim_time, chirp_stim, color='blue', label='刺激波形')
    plt.title(f'照射刺激波形 ')
    plt.xlabel('時間 [秒]')
    plt.ylabel('刺激強度')
    plt.grid(True)
    plt.legend(loc='upper right')

    # 下段: 細胞応答
    plt.subplot(2, 1, 2)
    plt.plot(resp_time, selected_roi_response, color='red', label=f'細胞応答 ({roi_label})')
    plt.title(f'細胞応答データ ({roi_label})')
    plt.xlabel('時間 [秒]')
    plt.ylabel('正規化蛍光強度 (ΔF/F) [Gult]')
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # --- 5. 参考: 重ね合わせプロット ---
    # 両方のデータの時間軸を合わせてプロットします
    plt.figure(figsize=(12, 6))
    
    # 単純な重ね合わせ（サンプリングレートが同じ場合）
    if stim_sampling_rate == resp_sampling_rate:
        # 短い方の長さに合わせる
        min_len = min(len(stim_time), len(resp_time))
        plt.plot(stim_time[:min_len], chirp_stim[:min_len], color='blue', alpha=0.6, label='刺激')
        plt.plot(resp_time[:min_len], selected_roi_response[:min_len], color='red', alpha=0.8, label='応答')
    else:
        # サンプリングレートが異なる場合、応答データを刺激データの時間に合わせる（補間）
        f = interp1d(resp_time, selected_roi_response, kind='linear', fill_value="extrapolate")
        resampled_response = f(stim_time)
        plt.plot(stim_time, chirp_stim, color='blue', alpha=0.6, label='刺激')
        plt.plot(stim_time, resampled_response, color='red', alpha=0.8, label='応答 (リサンプリング済)')

    plt.title('刺激と応答の重ね合わせ')
    plt.xlabel('時間 [秒]')
    plt.ylabel('強度 / 蛍光')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()