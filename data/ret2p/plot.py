import numpy as np
import matplotlib.pyplot as plt

# --- 1. 設定 ---
# ファイル名
stim_filename = 'data/ret2p/chirp_stimulus.txt'
# 応答データファイルのフォーマット (反復回数は %d で指定)
resp_filename_format = 'data/ret2p/response_data_repeat_%d.txt'

# 読み込む反復回数を指定 (MATLABでこのファイル名で保存した場合)
# 例: MATLABで 'response_data_repeat_1.txt' として保存したなら、1 を指定
target_repeat_to_load = 1 

# プロットしたいROI (細胞) のインデックスを指定
# Pythonは0-based indexなので、最初の細胞なら 0、10番目の細胞なら 9
target_roi_index = 0 

# サンプリングレート (Hz)
stim_sampling_rate = 1000  # chirp_stim は1000Hz (1kHz)
resp_sampling_rate = 64    # lchirp_avg (応答データ) は64Hz

# --- 2. chirp_stimulus.txt の読み込みとプロット ---
try:
    chirp_stim = np.loadtxt(stim_filename, delimiter='\t')
    stim_time = np.arange(0, len(chirp_stim)) / stim_sampling_rate

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1) # 2行1列のグラフの1番目
    plt.plot(stim_time, chirp_stim, color='blue', label='Chirp Stimulus')
    plt.title('Chirp Stimulus Waveform ({} samples)'.format(len(chirp_stim)))
    plt.xlabel('Time (s)')
    plt.ylabel('Stimulus Intensity')
    plt.grid(True)
    plt.legend()

except FileNotFoundError:
    print(f"エラー: {stim_filename} が見つかりません。ファイルパスを確認してください。")
    plt.figure() # エラーでもグラフは作成し、後続のプロットのためにsubplot数を設定
    plt.subplot(2, 1, 1)

# --- 3. response_data_repeat_X.txt の読み込みとプロット ---
# 指定した反復回数のファイル名を生成
resp_current_filename = resp_filename_format % target_repeat_to_load

try:
    # 応答データは [サンプル数 x ROI数] の2次元配列
    response_data = np.loadtxt(resp_current_filename, delimiter='\t')
    
    # 選択されたROIのデータを抽出
    if target_roi_index < response_data.shape[1]:
        selected_roi_response = response_data[:, target_roi_index]
    else:
        print(f"警告: 指定されたROIインデックス {target_roi_index} は存在しません。")
        print(f"利用可能なROIの数は {response_data.shape[1]} です。最初のROIをプロットします。")
        selected_roi_response = response_data[:, 0]
        target_roi_index = 0 # 最初のROIに設定
        
    resp_time = np.arange(0, len(selected_roi_response)) / resp_sampling_rate

    plt.subplot(2, 1, 2) # 2行1列のグラフの2番目
    plt.plot(resp_time, selected_roi_response, color='red', label=f'Response (Repeat {target_repeat_to_load}, ROI {target_roi_index})')
    plt.title(f'Cellular Response to Chirp Stimulus (Repeat {target_repeat_to_load}, ROI {target_roi_index})')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Fluorescence (a.u.)')
    plt.grid(True)
    plt.legend()

except FileNotFoundError:
    print(f"エラー: {resp_current_filename} が見つかりません。ファイルパスを確認してください。")
    plt.subplot(2, 1, 2) # エラーでもグラフは作成

plt.tight_layout() # サブプロット間のスペースを調整
plt.show()

# --- 参考: 刺激と応答を重ねてプロットする場合の注意 ---
# サンプリングレートが異なるため、直接重ねると時間軸がずれるか、プロットが粗くなります。
# 適切に重ねるには、どちらかのデータをリサンプリングする必要があります。
# 例: 応答データを刺激データと同じレートにアップサンプリング
if 'stim_time' in locals() and 'selected_roi_response' in locals():
    from scipy.interpolate import interp1d

    f = interp1d(resp_time, selected_roi_response, kind='linear', fill_value="extrapolate")
    resampled_response = f(stim_time)

    plt.figure(figsize=(12, 6))
    plt.plot(stim_time, chirp_stim, color='blue', alpha=0.7, label='Chirp Stimulus (1000Hz)')
    plt.plot(stim_time, resampled_response, color='red', alpha=0.7, label=f'Resampled Response (Repeat {target_repeat_to_load}, ROI {target_roi_index})')
    plt.title('Chirp Stimulus and Resampled Response Overlaid')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity / Normalized Fluorescence')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()