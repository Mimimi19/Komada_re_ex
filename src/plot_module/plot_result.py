# src/plot_result.py
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 日本語フォント対応
try:
    import japanize_matplotlib
except ImportError:
    pass

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_axis(ax, title, ylabel, xlabel='Time (s)'):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(which='major', linestyle='--', alpha=0.5)

def process_plot(config_path, result_dir, target_tau=None):
    """
    main.py から呼び出される描画処理の本体
    """
    print(f"\n--- Plotting Results ---")
    print(f"Config: {config_path}")
    print(f"Result Dir: {result_dir}")

    # コンフィグ読み込み
    try:
        cfg_data = load_yaml(config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return

    # データ読み込み (config内のパスは絶対パスか、プロジェクトルートからの相対パスと仮定)
    # main.pyから呼ばれる場合、パスの解決に注意が必要
    # ここでは config.yaml の input_file を信頼するが、
    # 一時ファイルを使用している場合は main.py が正しい config を渡す必要がある
    
    # 簡易的に、result_dir 内にある predict.txt と params.txt を使用してプロットする
    predict_path = os.path.join(result_dir, "predict.txt")
    
    if not os.path.exists(predict_path):
        print(f"Error: predict.txt not found in {result_dir}")
        return

    # 予測データの読み込み
    prediction = np.genfromtxt(predict_path)

    # 実測データの読み込み（Configから）
    # プロジェクトルート基準でパスを解決
    # 注意: Hydra実行時はcwdが変わるが、このスクリプトは絶対パスで扱うのが安全
    if os.path.isabs(cfg_data['input_file']):
        output_file = cfg_data['output_file']
    else:
        # 相対パスの場合、configファイルの場所ではなく、実行時のルート(想定)から探す
        # ここでは main.py と同じ階層にある config/data/... を想定
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_file = os.path.join(project_root, cfg_data['output_file'])

    if os.path.exists(output_file):
        raw_output = np.genfromtxt(output_file)
        # 正規化
        output_data = raw_output / np.max(np.abs(raw_output))
    else:
        print(f"Warning: Measured output file not found at {output_file}. Plotting prediction only.")
        output_data = None

    dt = cfg_data.get('dt', 0.0002)
    time_axis = np.arange(len(prediction)) * dt

    # プロット作成
    plt.figure(figsize=(10, 6))
    if output_data is not None:
        # 長さを合わせる
        min_len = min(len(output_data), len(prediction))
        plt.plot(time_axis[:min_len], output_data[:min_len], color='black', alpha=0.5, label='Measured')
        plt.plot(time_axis[:min_len], prediction[:min_len], color='red', alpha=0.8, label='Predicted')
    else:
        plt.plot(time_axis, prediction, color='red', label='Predicted')

    plt.title(f"Result Plot\nDir: {os.path.basename(result_dir)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Response")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = os.path.join(result_dir, "auto_plot_result.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    # コマンドライン実行時の処理
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to data config yaml")
    parser.add_argument("result_dir", help="Path to result directory")
    args = parser.parse_args()
    
    process_plot(args.config_path, args.result_dir)