#LowPass_refactored_subplots_jp.py
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ------------------- 日本語対応のための設定 -------------------
# Matplotlibで日本語を使用するための設定
# ご自身の環境にインストールされている日本語フォントを指定してください。
# 例:
# Windows: 'Yu Gothic', 'Meiryo', 'MS Gothic'
# macOS: 'Hiragino Sans'
# Linux: 'IPAexGothic', 'TakaoPGothic' など
try:
    plt.rcParams['font.family'] = 'Hiragino Sans'  # 例: macOSのフォント
    plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化けを防ぐ
except Exception as e:
    print(f"警告: 日本語フォントの設定に失敗しました。グラフの日本語が文字化けする可能性があります。({e})")
    print("スクリプト内の 'font.family' をお使いのPCにインストールされているフォント名に変更してください。")
# -----------------------------------------------------------


def process_data(input_filename, output_filename, sampling_rate, cutoff_freq, title_suffix, color_original, color_filtered):
    """
    データファイルを読み込み、ローパスフィルター処理を行い、結果を保存・描画する関数。
    グラフは変換前と変換後を上下2段で表示する。
    
    Args:
        input_filename (str): 入力データファイル名
        output_filename (str): 出力データファイル名
        sampling_rate (float): サンプリング周波数
        cutoff_freq (float): カットオフ周波数
        title_suffix (str): グラフタイトルに使用する接尾辞 (例: "data1")
        color_original (str): 元データのグラフ色
        color_filtered (str): フィルター後データのグラフ色
    """
    # --- データの読み込み ---
    try:
        data = np.genfromtxt(input_filename)
    except FileNotFoundError:
        print(f"エラー: '{input_filename}' が見つかりません。")
        print("コードを実行する前に、データファイルを正しいディレクトリに配置してください。")
        return # エラーが発生した場合は関数を終了

    # --- 時間軸の作成 ---
    time = np.arange(len(data)) / sampling_rate

    # --- フーリエ変換とフィルター処理 ---
    fft_data = np.fft.fft(data)
    N = len(data)
    freq = np.fft.fftfreq(N, d=1/sampling_rate)
    
    fft_data_filtered = fft_data.copy()
    fft_data_filtered[np.abs(freq) > cutoff_freq] = 0
    
    filtered_data = np.real(np.fft.ifft(fft_data_filtered))

    # --- 処理後のデータをファイルに保存 ---
    np.savetxt(output_filename, filtered_data)
    print(f"フィルター処理後のデータが '{output_filename}' に保存されました。")

    # --- グラフ描画処理 ---

    # 1つの図に上下2つのグラフを作成 (2行1列の意)。x軸を共有する
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'ローパスフィルター処理 ({title_suffix})', fontsize=20) # 図全体のタイトル

    # 上段: 元データのグラフ
    axes[0].plot(time, data, color_original)
    axes[0].set_title('オリジナルデータ')
    axes[0].set_ylabel('Input[pA]', fontsize=16)
    axes[0].grid(True)

    # 下段: フィルター処理後のグラフ
    axes[1].plot(time, filtered_data, color_filtered)
    axes[1].set_title(f'フィルター処理後のデータ (Low-pass < {cutoff_freq}Hz)')
    axes[1].set_xlabel('Time[s]', fontsize=16)
    axes[1].set_ylabel('Filtered Input[pA]', fontsize=16)
    axes[1].grid(True)

    # 両方のグラフに共通のx軸の範囲を設定
    plt.xlim(0, 16)
    
    # グラフのレイアウトを自動調整
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 全体のタイトルと重ならないように調整

    # グラフを表示
    plt.show()


def main():
    """
    メイン処理
    """
    # 共通のパラメータを設定
    sampling_rate = 5000.0
    cutoff_frequency = 60.0

    # --- データ1の処理を呼び出し ---
    process_data(
        input_filename="cb1/wn_0.0002s.txt",
        output_filename="cb1/U60wn_0.0002s.txt",
        sampling_rate=sampling_rate,
        cutoff_freq=cutoff_frequency,
        title_suffix="cb1 Input",
        color_original="b",
        color_filtered="r"
    )

    # --- データ2の処理を呼び出し ---
    process_data(
        input_filename="cb2/wn.txt",
        output_filename="cb2/U60wn.txt",
        sampling_rate=sampling_rate,
        cutoff_freq=cutoff_frequency,
        title_suffix="cb2 Input",
        color_original="b",
        color_filtered="r"
    )
    
    process_data(
        input_filename="cb1/average80-86.txt",
        output_filename="cb1/U60average80-86.txt",
        sampling_rate=sampling_rate,
        cutoff_freq=cutoff_frequency,
        title_suffix="cb1 Output",
        color_original="c",
        color_filtered="m"
    )
    
    process_data(
        input_filename="cb2/average82-88.txt",
        output_filename="cb2/U60average82-88.txt",
        sampling_rate=sampling_rate,
        cutoff_freq=cutoff_frequency,
        title_suffix="cb2 Output",
        color_original="c",
        color_filtered="m"
    )

# このスクリプトが直接実行された場合にmain関数を呼び出す
if __name__ == "__main__":
    main()