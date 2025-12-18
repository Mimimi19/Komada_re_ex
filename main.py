# main.py
import os
import sys
import datetime
import numpy as np
import hydra
from omegaconf import DictConfig
import subprocess
from pathlib import Path
import shutil

# コンポーネントへのパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from components.segment_manager import SegmentManager
import plot_result

def get_save_root_name(data_name):
    """データ名に応じて保存先ディレクトリ名を決定"""
    if "cb1" in data_name: return "Baccus_cb1"
    elif "cb2" in data_name: return "Baccus_cb2"
    elif "ret2p" in data_name: return "Baccus_ret2p"
    else: return f"Baccus_{data_name}"

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("=== Baccus Model Orchestrator Started ===")
    
    project_root = hydra.utils.get_original_cwd()
    data_name = cfg.data.name
    dt = cfg.data.get('dt', 0.0002) 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # 1. 保存先ルートディレクトリの決定
    # 例: scripts/results/Baccus_cb2/20251218_1744
    save_root_name = get_save_root_name(data_name)
    base_save_dir = Path(project_root) / "scripts" / "results" / save_root_name / timestamp
    base_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 目的関数のタイプ
    objective_type = cfg.hyper_params.get("objective_type", "hybrid")
    print(f"Objective Type: {objective_type}")
    print(f"Target Directory: {base_save_dir}")

    # データの読み込み
    input_file_path = Path(project_root) / cfg.data.input_file
    output_file_path = Path(project_root) / cfg.data.output_file
    raw_input = np.genfromtxt(input_file_path)
    raw_output = np.genfromtxt(output_file_path)

    # segment指定の取得
    segment_duration = cfg.get("segment")

    # === MODE A: 全文処理 (Segment指定なし) ===
    if segment_duration is None:
        print("\n[Mode: Full Data Processing]")
        
        # 全文処理の場合はルートディレクトリ直下に結果を保存
        cmd = [
            "uv", "run", "python", "src/model/BaccusModel.py",
            f"data.input_file={input_file_path}",
            f"data.output_file={output_file_path}",
            f"data.name={data_name}", 
            f"data.dt={dt}",
            f"hydra.run.dir={base_save_dir}", # ルートに保存
            f"hyper_params.objective_type={objective_type}"
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=project_root)
            # プロット
            result_config = base_save_dir / ".hydra" / "config.yaml"
            if result_config.exists():
                plot_result.process_plot(str(result_config), str(base_save_dir))
        except subprocess.CalledProcessError as e:
            print(f"Execution failed: {e}")

    # === MODE B: セグメント処理 (Segment指定あり) ===
    else:
        segment_sec = float(segment_duration)
        print(f"\n[Mode: Segment Processing] Duration = {segment_sec} sec")
        
        # 2. temp_files ディレクトリの作成 (ルート直下)
        # scripts/results/Baccus_cb2/20251218_1744/temp_files
        temp_dir = base_save_dir / "temp_files"
        temp_dir.mkdir(exist_ok=True)

        warmup_sec = cfg.hyper_params.tau
        
        # 分割ファイルの生成
        manager = SegmentManager(project_root, dt)
        segments = manager.create_segments(raw_input, raw_output, segment_sec, warmup_sec, temp_dir)
        
        collected_params = []

        for seg in segments:
            seg_id = seg['id']
            trim_sec = seg['trim_sec']
            print(f"\n--- Processing Segment {seg_id}/{len(segments)} ---")
            print(f"{seg['info']}")
            
            # 3. 各セグメントの保存先ディレクトリ
            # scripts/results/Baccus_cb2/20251218_1744/1_segment
            seg_run_dir = base_save_dir / f"{seg_id}_segment"
            
            # BaccusModelの実行
            # hydra.run.dir を seg_run_dir に指定することで、
            # state/, epochs/, params.txt などがこのフォルダ内に自動生成される
            cmd = [
                "uv", "run", "python", "src/model/BaccusModel.py",
                f"data.input_file={seg['input_path']}",   # temp_files内のパス
                f"data.output_file={seg['output_path']}", # temp_files内のパス
                f"data.name={data_name}", 
                f"data.dt={dt}",
                f"hydra.run.dir={seg_run_dir}",
                f"hyper_params.trim_I_seconds={trim_sec}",
                f"hyper_params.objective_type={objective_type}", 
                "optimization.popsize=300",
                "optimization.maxiter=200"
            ]
            
            try:
                subprocess.run(cmd, check=True, cwd=project_root)
                
                params_file = seg_run_dir / "params.txt"
                if params_file.exists():
                    params = np.genfromtxt(params_file)
                    collected_params.append(params)
                    
                    # プロット (segmentフォルダ内に保存)
                    result_config = seg_run_dir / ".hydra" / "config.yaml"
                    plot_result.process_plot(str(result_config), str(seg_run_dir))

            except subprocess.CalledProcessError:
                print(f"Skipping segment {seg_id} due to error.")
                continue

        # 4. 中央値計算と最終検証
        if collected_params:
            print("\n--- Calculating Median Parameters ---")
            median_params = np.median(collected_params, axis=0)
            median_save_path = base_save_dir / "median_params.txt"
            np.savetxt(median_save_path, median_params, fmt='%.6f')
            
            print("--- Running Final Verification with Full Data ---")
            
            # 最終結果はルートディレクトリ (base_save_dir) に保存する
            # これにより state/, epochs/ がルート直下に作られる
            cmd_final = [
                "uv", "run", "python", "src/model/BaccusModel.py",
                f"data.input_file={input_file_path}",
                f"data.output_file={output_file_path}",
                f"data.name={data_name}",
                f"data.dt={dt}",
                f"hydra.run.dir={base_save_dir}",
                f"hyper_params.objective_type={objective_type}"
            ]
            subprocess.run(cmd_final, check=True, cwd=project_root)
            
            result_config_final = base_save_dir / ".hydra" / "config.yaml"
            if result_config_final.exists():
                plot_result.process_plot(str(result_config_final), str(base_save_dir))
            
            print(f"\nAll Completed. Results: {base_save_dir}")
        else:
            print("No parameters collected.")

if __name__ == "__main__":
    main()