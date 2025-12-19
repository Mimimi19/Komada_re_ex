import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import glob
import re
import yaml

# 日本語フォント対応 (あれば)
try:
    import japanize_matplotlib
except ImportError:
    pass

def natural_sort_key(s):
    """文字列内の数字を数値としてソートするためのキー生成関数"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def find_dt(target_dir):
    """
    .hydra/config.yaml から dt を取得を試みる。
    見つからない場合はデフォルト値を返す。
    """
    # 探索パターンの作成
    search_paths = [
        os.path.join(target_dir, "**", ".hydra", "config.yaml"), # サブディレクトリ内
        os.path.join(target_dir, ".hydra", "config.yaml")        # ルート直下
    ]
    
    for pattern in search_paths:
        configs = glob.glob(pattern, recursive=True)
        if configs:
            try:
                cfg = load_yaml(configs[0])
                if 'data' in cfg and 'dt' in cfg['data']:
                    return float(cfg['data']['dt'])
                if 'hyper_params' in cfg and 'dt' in cfg['hyper_params']:
                    return float(cfg['hyper_params']['dt'])
            except Exception:
                pass
            
    print("Warning: dt not found in config. Using default 0.0002")
    return 0.0002

def process_segment_plot(target_dir):
    print(f"Target Directory: {target_dir}")
    
    # 1. セグメントフォルダの特定
    segment_pattern = os.path.join(target_dir, "*_segment")
    segment_dirs = glob.glob(segment_pattern)
    
    if not segment_dirs:
        print(f"Error: No segment folders found in {target_dir}")
        return

    # ID順にソート (1, 2, 10 と並ぶように)
    segment_dirs.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    # dtの取得
    dt = find_dt(target_dir)
    print(f"Using sampling rate dt = {dt} s")

    num_segments = len(segment_dirs)
    print(f"Found {num_segments} segments.")

    # 2. プロットの準備
    # 行数 = セグメント数, 列数 = 2 (Input, Output)
    fig, axes = plt.subplots(num_segments, 2, figsize=(15, 3 * num_segments), sharex='row')
    if num_segments == 1:
        axes = axes.reshape(1, -1)

    # 3. 各セグメントの処理ループ
    for i, seg_dir in enumerate(segment_dirs):
        folder_name = os.path.basename(seg_dir)
        # フォルダ名からIDを抽出 (例: "1_segment" -> 1)
        match = re.match(r"(\d+)_segment", folder_name)
        if not match:
            continue
        
        seg_id = match.group(1)
        
        # パスの構築
        # 入出力: target_dir/temp_files/{id}_input.txt
        input_path = os.path.join(target_dir, "temp_files", f"{seg_id}_input.txt")
        output_path = os.path.join(target_dir, "temp_files", f"{seg_id}_output.txt")
        # モデル予測: target_dir/{id}_segment/state/W_state.txt
        model_state_path = os.path.join(seg_dir, "state", "W_state.txt")
        
        # データの読み込み
        if not (os.path.exists(input_path) and os.path.exists(output_path)):
            print(f"Warning: Input/Output files not found for segment {seg_id}")
            continue
            
        u_stim = np.genfromtxt(input_path)
        y_resp = np.genfromtxt(output_path)
        
        # モデル予測データの取得
        if os.path.exists(model_state_path):
            # W_state を読み込み、符号を反転 (電流 W は負の応答)
            w_state = np.genfromtxt(model_state_path)
            y_pred = -1.0 * w_state
        else:
            print(f"Warning: Model state (W_state.txt) not found for segment {seg_id}")
            y_pred = np.zeros_like(y_resp)

        # 長さの調整
        min_len = min(len(u_stim), len(y_resp), len(y_pred))
        u_stim = u_stim[:min_len]
        y_resp = y_resp[:min_len]
        y_pred = y_pred[:min_len]
        
        # 時間軸
        time_axis = np.arange(min_len) * dt

        # --- 正規化 (比較のため) ---
        # 実測値を最大値で正規化（BaccusModelの処理に合わせる）
        if np.max(np.abs(y_resp)) > 1e-9:
            y_resp_norm = y_resp / np.max(np.abs(y_resp))
        else:
            y_resp_norm = y_resp

        # スコア計算
        if np.std(y_pred) > 1e-9 and np.std(y_resp_norm) > 1e-9:
            corr, _ = spearmanr(y_resp_norm, y_pred)
        else:
            corr = 0.0

        # --- プロット: Left (Input) ---
        ax_in = axes[i, 0]
        ax_in.plot(time_axis, u_stim, color='gray', linewidth=1.0)
        ax_in.set_ylabel('Stimulus')
        ax_in.set_title(f"Segment {seg_id}: Input", fontsize=10)
        ax_in.grid(True, linestyle=':', alpha=0.6)

        # --- プロット: Right (Output Comparison) ---
        ax_out = axes[i, 1]
        ax_out.plot(time_axis, y_resp_norm, color='black', alpha=0.5, label='Measured', linewidth=1.5)
        ax_out.plot(time_axis, y_pred, color='red', alpha=0.8, label='Model', linewidth=1.5)
        
        ax_out.set_ylabel('Response (Norm)')
        ax_out.set_title(f"Segment {seg_id}: Output (Spearman R={corr:.3f})", fontsize=10)
        ax_out.legend(loc='upper right', fontsize=8)
        ax_out.grid(True, linestyle=':', alpha=0.6)
        
        # 最後の行だけX軸ラベルを表示
        if i == num_segments - 1:
            ax_in.set_xlabel('Time (s)')
            ax_out.set_xlabel('Time (s)')

    plt.tight_layout()
    
    # 保存
    save_filename = "segment_analysis_summary.png"
    save_path = os.path.join(target_dir, save_filename)
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved summary plot to: {save_path}")
    
    # 画像を表示せずに閉じる (サーバー環境等を考慮)
    plt.close()

if __name__ == "__main__":
    # =================================================================
    # ▼▼▼ 設定エリア: データの場所を指定してください ▼▼▼
    # =================================================================
    
    # 例: 
    # TARGET_DIR = "scripts/segments/ret2p/20251218_1906"
    
    # 引数があればそれを使用、なければデフォルト（安全策）
    if len(sys.argv) > 1:
        TARGET_DIR = sys.argv[1]
    else:
        # デフォルト値（必要に応じて書き換えてください）
        TARGET_DIR = "scripts/results/Baccus_ret2p/20251218_1906" # 例

    # ディレクトリの存在確認
    if not os.path.exists(TARGET_DIR):
        print(f"Error: Directory not found: {TARGET_DIR}")
        print("Usage: uv run src/segment_plot.py <path_to_result_dir>")
    else:
        process_segment_plot(TARGET_DIR)