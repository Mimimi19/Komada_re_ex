#LowPass_refactored_overlay_jp_pdf.py
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------- 日本語対応のための設定 -------------------
try:
    plt.rcParams['font.family'] = 'Hiragino Sans'
    plt.rcParams['axes.unicode_minus'] = False
    # PDF保存時の日本語文字化け対策
    plt.rcParams['pdf.fonttype'] = 42
except Exception as e:
    print(f"警告: 日本語フォントの設定に失敗しました。({e})")
# -----------------------------------------------------------


def process_data(input_filename, output_filename, sampling_rate, cutoff_freq, title_suffix, color_original, color_filtered):
    """
    データファイルを読み込み、フィルター処理を行い、結果をテキストファイルとPDFグラフで保存・描画する関数。
    グラフは処理前と処理後を重ねて表示する。
    """
    # --- データの読み込み ---
    try:
        data = np.genfromtxt(input_filename)
    except FileNotFoundError:
        print(f"エラー: '{input_filename}' が見つかりません。")
        return

    # --- 時間軸とフィルター処理 ---
    time = np.arange(len(data)) / sampling_rate
    fft_data = np.fft.fft(data)
    N = len(data)
    freq = np.fft.fftfreq(N, d=1/sampling_rate)
    fft_data_filtered = fft_data.copy()
    fft_data_filtered[np.abs(freq) > cutoff_freq] = 0
    filtered_data = np.real(np.fft.ifft(fft_data_filtered))

    # --- テキストファイルに保存 ---
    np.savetxt(output_filename, filtered_data)
    print(f"フィルター処理後のデータが '{output_filename}' に保存されました。")

    # --- グラフ描画処理 (ここから変更) ---

    # 1つの図と1つのグラフ領域を作成
    fig, ax = plt.subplots(figsize=(12, 6))

    # オリジナルデータとフィルター処理後のデータを重ねて描画
    ax.plot(time, data, color=color_original, alpha=0.5, label='オリジナルデータ')
    ax.plot(time, filtered_data, color=color_filtered, alpha=0.5, label=f'フィルター処理後 (< {cutoff_freq}Hz)')

    # グラフのタイトルとラベル設定
    ax.set_title(f'フィルター処理前後の比較 ({title_suffix})', fontsize=20)
    ax.set_xlabel('Time[s]', fontsize=16)
    ax.set_ylabel('Input[pA]', fontsize=16)
    ax.set_xlim(0, 16)
    ax.grid(True)
    ax.legend() # 凡例を表示

    plt.tight_layout()
    
    # PDFファイル名を生成
    pdf_filename = os.path.splitext(output_filename)[0] + ".pdf"
    
    # グラフをPDFファイルとして保存
    plt.savefig(pdf_filename)
    print(f"グラフが '{pdf_filename}' に保存されました。")
    
    # グラフを画面に表示
    plt.show()

    # メモリ解放のために図を閉じる
    plt.close(fig)


def main():
    """
    メイン処理
    """
    sampling_rate = 5000.0
    cutoff_frequency = 60.0

    # 処理するファイルのリスト
    file_list = [
        {"input": "cb1/wn_0.0002s.txt", "output": "cb1/U60wn_0.0002s.txt", "suffix": "cb1 Input", "colors": ("b", "r")},
        {"input": "cb2/wn.txt", "output": "cb2/U60wn.txt", "suffix": "cb2 Input", "colors": ("b", "r")},
        {"input": "cb1/average80-86.txt", "output": "cb1/U60average80-86.txt", "suffix": "cb1 Output", "colors": ("c", "m")},
        {"input": "cb2/average82-88.txt", "output": "cb2/U60average82-88.txt", "suffix": "cb2 Output", "colors": ("c", "m")}
    ]

    for f in file_list:
        process_data(
            input_filename=f["input"],
            output_filename=f["output"],
            sampling_rate=sampling_rate,
            cutoff_freq=cutoff_frequency,
            title_suffix=f["suffix"],
            color_original=f["colors"][0],
            color_filtered=f["colors"][1]
        )

if __name__ == "__main__":
    main()