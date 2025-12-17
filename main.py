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
# src/components/segment_manager.py をインポート
from components.segment_manager import SegmentManager
# src/plot_result.py をインポート
import plot_result

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("=== Baccus Model Orchestrator Started ===")
    
    project_root = hydra.utils.get_original_cwd()
    data_name = cfg.data.name
    dt = cfg.data.get('dt', 0.0002) 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # 目的関数のタイプを取得 (configから)
    objective_type = cfg.hyper_params.get("objective_type", "hybrid")
    print(f"Objective Type: {objective_type}")

    # データの読み込み
    input_file_path = Path(project_root) / cfg.data.input_file
    output_file_path = Path(project_root) / cfg.data.output_file
    
    print(f"Loading Data: {input_file_path}")
    raw_input = np.genfromtxt(input_file_path)
    raw_output = np.genfromtxt(output_file_path)

    # segment指定の取得
    segment_duration = cfg.get("segment")

    # === MODE A: 全文処理 (Segment指定なし) ===
    if segment_duration is None:
        print("\n[Mode: Full Data Processing]")
        
        save_dir = Path(project_root) / "scripts" / "results" / data_name / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # BaccusModelに渡す引数を構築
        cmd = [
            "uv", "run", "python", "src/model/BaccusModel.py",
            f"data.input_file={input_file_path}",
            f"data.output_file={output_file_path}",
            f"hydra.run.dir={save_dir}",
            f"hyper_params.objective_type={objective_type}" # 目的関数を渡す
        ]
        
        try:
            print(f"Running full optimization... Output: {save_dir}")
            subprocess.run(cmd, check=True, cwd=project_root)
            
            # プロット作成
            result_config = save_dir / ".hydra" / "config.yaml"
            if result_config.exists():
                plot_result.process_plot(str(result_config), str(save_dir))
        except subprocess.CalledProcessError as e:
            print(f"Execution failed: {e}")

    # === MODE B: セグメント処理 (Segment指定あり) ===
    else:
        segment_sec = float(segment_duration)
        print(f"\n[Mode: Segment Processing] Duration = {segment_sec} sec")
        
        base_save_dir = Path(project_root) / "scripts" / "segments" / data_name / timestamp
        base_save_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = base_save_dir / "temp_files"
        temp_dir.mkdir(exist_ok=True)

        warmup_sec = cfg.hyper_params.tau
        
        manager = SegmentManager(project_root, dt)
        segments = manager.create_segments(raw_input, raw_output, segment_sec, warmup_sec, temp_dir)
        
        collected_params = []

        for seg in segments:
            seg_id = seg['id']
            trim_sec = seg['trim_sec']
            print(f"\n--- Processing Segment {seg_id}/{len(segments)} ---")
            print(f"{seg['info']}")
            
            seg_run_dir = base_save_dir / f"{seg_id}_segment"
            
            cmd = [
                "uv", "run", "python", "src/model/BaccusModel.py",
                f"data.input_file={seg['input_path']}",
                f"data.output_file={seg['output_path']}",
                f"hydra.run.dir={seg_run_dir}",
                f"hyper_params.trim_I_seconds={trim_sec}",
                f"hyper_params.objective_type={objective_type}", # 目的関数を渡す
                "optimization.popsize=15",
                "optimization.maxiter=50"
            ]
            
            try:
                subprocess.run(cmd, check=True, cwd=project_root)
                
                params_file = seg_run_dir / "params.txt"
                if params_file.exists():
                    params = np.genfromtxt(params_file)
                    collected_params.append(params)
                    
                    # --- 解析用ファイルの保存 ---
                    # 1. 使用した刺激と応答 (Warmup含む実際にモデルに入力された範囲)
                    shutil.copy(seg['input_path'], seg_run_dir / f"{seg_id}_stimulus.txt")
                    shutil.copy(seg['output_path'], seg_run_dir / f"{seg_id}_response.txt")
                    
                    # 2. 最終パラメータの複製
                    # formatは 'epoch_600_params.txt' と同じく、数値が1列に並んだテキストファイルです
                    shutil.copy(params_file, seg_run_dir / f"{seg_id}_params.txt")
                    
                    # 3. 予測結果
                    predict_file = seg_run_dir / "predict.txt"
                    if predict_file.exists():
                        shutil.copy(predict_file, seg_run_dir / f"{seg_id}_predict.txt")

                    print(f"Saved: {seg_id}_params.txt, {seg_id}_stimulus.txt, etc.")

                    # プロット
                    result_config = seg_run_dir / ".hydra" / "config.yaml"
                    plot_result.process_plot(str(result_config), str(seg_run_dir))

            except subprocess.CalledProcessError:
                print(f"Skipping segment {seg_id} due to error.")
                continue

        # 中央値計算と全体検証
        if collected_params:
            print("\n--- Calculating Median Parameters ---")
            median_params = np.median(collected_params, axis=0)
            median_save_path = base_save_dir / "median_params.txt"
            np.savetxt(median_save_path, median_params, fmt='%.6f') # フォーマット指定
            
            print("--- Running Final Verification with Full Data ---")
            final_run_dir = base_save_dir / "final_verification"
            
            cmd_final = [
                "uv", "run", "python", "src/model/BaccusModel.py",
                f"data.input_file={input_file_path}",
                f"data.output_file={output_file_path}",
                f"hydra.run.dir={final_run_dir}",
                f"hyper_params.objective_type={objective_type}"
            ]
            subprocess.run(cmd_final, check=True, cwd=project_root)
            
            result_config_final = final_run_dir / ".hydra" / "config.yaml"
            if result_config_final.exists():
                plot_result.process_plot(str(result_config_final), str(final_run_dir))
            
            print(f"\nAll Completed. Results: {base_save_dir}")
        else:
            print("No parameters collected.")

if __name__ == "__main__":
    main()