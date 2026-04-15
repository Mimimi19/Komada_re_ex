# src/model/SegmentLearning.py
import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve
import mlflow

# --- パス調整 ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/model
src_dir = os.path.dirname(current_dir) # src
if src_dir not in sys.path:
    sys.path.append(src_dir)

# 自作モジュールのインポート
import components.L_LNK as L_LNK
import components.N_LNK as N_LNK
import components.K_baccus as K_LNK

# 目的関数モジュールの動的インポート用
import importlib

class SegmentLearningModel:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.J = cfg.hyper_params.J
        self.dt = cfg.data.dt  # 0.0002 s
        self.tau = cfg.hyper_params.tau 
        
        # データの読み込み
        input_file = hydra.utils.to_absolute_path(cfg.data.input_file)
        output_file = hydra.utils.to_absolute_path(cfg.data.output_file)

        # データの正規化
        raw_input = np.genfromtxt(input_file)
        self.Input_full = (raw_input - np.mean(raw_input)) / np.std(raw_input)
        
        raw_output = np.genfromtxt(output_file)
        self.Output_full = raw_output / np.max(np.abs(raw_output))
        
        # 学習に使用するデータ（動的に変更）
        self.current_input = self.Input_full
        self.current_output = self.Output_full

        # 保存先のベースディレクトリ作成 (scripts/segments/{data_name})
        self.project_root = hydra.utils.get_original_cwd()
        self.data_name = cfg.data.name  # 'cb1', 'ret2p-1' etc.
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        
        self.base_save_path = Path(self.project_root) / "scripts" / "segments" / self.data_name
        self.base_save_path.mkdir(parents=True, exist_ok=True)

        # MLflow設定
        mlflow.set_experiment("Baccus_Segment_Learning")
        
        # 目的関数の選択
        obj_type = cfg.hyper_params.objective_type
        if obj_type == "spearman":
             import components.objectives.spearman as obj_module
        elif obj_type == "hybrid":
             import components.objectives.hybrid as obj_module
        else:
             try:
                 obj_module = importlib.import_module(f"components.objectives.{obj_type}")
             except ImportError:
                 print(f"Warning: Objective type '{obj_type}' not found. Using hybrid.")
                 import components.objectives.hybrid as obj_module
                 
        self.objective_func = obj_module.calculate

    def _calculate_steady_state(self, u_val, ka, kfi, kfr, ksi, ksr):
        if kfi > 1e-9: A_ratio = (ka * u_val) / kfi
        else: A_ratio = 0.0
        if kfr > 1e-9: I1_ratio = (ka * u_val) / kfr
        else: I1_ratio = 0.0
        if kfr > 1e-9 and ksr > 1e-9:
            I2_ratio = (ksi * I1_ratio) / (ksr * u_val) if u_val > 1e-6 else 0.0
        else: I2_ratio = 0.0

        total = 1.0 + A_ratio + I1_ratio + I2_ratio
        return (1.0/total), (A_ratio/total), (I1_ratio/total), (I2_ratio/total)

    def _simulate_and_save(self, x, input_data, target_output, save_dir: Path, label="result"):
        """
        パラメータxを用いてシミュレーションを行い、結果を指定ディレクトリに保存するヘルパー関数
        """
        # パラメータ展開
        alphas = x[0:self.J]
        delta = x[self.J]
        a, kappa, b1, b2, ka = x[self.J+1:self.J+6]
        kfi, kfr, ksi, ksr = x[self.J+6:self.J+10]
        w_gain, w_decay = x[self.J+10:self.J+12]
        
        if not self.cfg.hyper_params.use_I2:
            ksi = 0.0; ksr = 0.0

        # Linear
        filter_points = int(self.tau / self.dt) + 1
        kernel, _ = L_LNK.main(alphas, delta, filter_points, self.dt, self.tau)
        g_full = fftconvolve(input_data, kernel, mode='full')
        shift_idx = int(self.tau / self.dt)
        g_t = g_full[shift_idx : shift_idx + len(input_data)]
        if np.std(g_t) > 1e-9: g_t = g_t / np.std(g_t)

        # Nonlinear
        u_t = N_LNK.main(g_t, a, kappa, b1, b2, ka)

        # Kinetic
        u_start = max(0.0, u_t[0])
        p_R, p_A, p_I1, p_I2 = self._calculate_steady_state(u_start, ka, kfi, kfr, ksi, ksr)
        if not self.cfg.hyper_params.use_I2: p_I1 += p_I2; p_I2 = 0.0

        _, _, _, _, W_state, _ = K_LNK.main(
            len(u_t), u_t, self.dt, 
            p_R, p_A, p_I1, p_I2, 
            ka, kfi, kfr, ksi, ksr, 
            w_gain, w_decay, label="save"
        )
        
        prediction = -1.0 * W_state
        min_len = min(len(target_output), len(prediction))
        
        # --- 保存処理 ---
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. パラメータ
        np.savetxt(save_dir / "params.txt", x)
        
        # 2. 予測データ
        np.savetxt(save_dir / "predict.txt", prediction)
        
        # 3. プロット
        plt.figure(figsize=(10, 5))
        plt.plot(target_output[:min_len], label='Target', alpha=0.6)
        plt.plot(prediction[:min_len], label='Predict', alpha=0.8, linestyle='--')
        plt.title(f"Simulation Result: {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "plot.png")
        plt.close() # メモリ解放
        
        return prediction

    def lnk_model(self, x):
        # パラメータ展開
        alphas = x[0:self.J]
        delta = x[self.J]
        a, kappa, b1, b2, ka = x[self.J+1:self.J+6]
        kfi, kfr, ksi, ksr = x[self.J+6:self.J+10]
        w_gain, w_decay = x[self.J+10:self.J+12]

        if not self.cfg.hyper_params.use_I2:
            ksi = 0.0; ksr = 0.0

        # Linear Filter
        filter_points = int(self.tau / self.dt) + 1
        kernel, _ = L_LNK.main(alphas, delta, filter_points, self.dt, self.tau)
        g_full = fftconvolve(self.current_input, kernel, mode='full')
        
        shift_idx = int(self.tau / self.dt)
        g_t = g_full[shift_idx : shift_idx + len(self.current_input)]
        
        if np.std(g_t) > 1e-9: g_t = g_t / np.std(g_t)

        # Nonlinear
        u_t = N_LNK.main(g_t, a, kappa, b1, b2, ka)

        # Kinetic
        u_start = max(0.0, u_t[0])
        p_R, p_A, p_I1, p_I2 = self._calculate_steady_state(u_start, ka, kfi, kfr, ksi, ksr)
        if not self.cfg.hyper_params.use_I2:
            p_I1 += p_I2; p_I2 = 0.0

        _, _, _, _, W_state, check = K_LNK.main(
            len(u_t), u_t, self.dt, 
            p_R, p_A, p_I1, p_I2, 
            ka, kfi, kfr, ksi, ksr, 
            w_gain, w_decay, label="opt"
        )

        if check == 0: return 5.0 

        # Evaluation
        model_out = -1.0 * W_state
        min_len = min(len(self.current_output), len(model_out))
        output_eval = self.current_output[:min_len]
        model_eval = model_out[:min_len]

        return self.objective_func(output_eval, model_eval)

    def callback(self, xk, convergence=None):
        score = self.lnk_model(xk)
        print(f"Current Score: {score:.5f}", end='\r')

    def run(self):
        bounds_dict = self.cfg.hyper_params.param_bounds
        bounds = [tuple(bounds_dict.LinearFilter.alphas) for _ in range(self.J)]
        bounds.append(tuple(bounds_dict.LinearFilter.delta))
        bounds.append(tuple(bounds_dict.Nonlinear.a))
        bounds.append(tuple(bounds_dict.Nonlinear.kappa))
        bounds.append(tuple(bounds_dict.Nonlinear.b1))
        bounds.append(tuple(bounds_dict.Nonlinear.b2))
        bounds.append(tuple(bounds_dict.Kinetics.ka))
        bounds.append(tuple(bounds_dict.Kinetics.kfi))
        bounds.append(tuple(bounds_dict.Kinetics.kfr))
        bounds.append(tuple(bounds_dict.Kinetics.ksi))
        bounds.append(tuple(bounds_dict.Kinetics.ksr))
        bounds.append(tuple(bounds_dict.Kinetics.w_gain))
        bounds.append(tuple(bounds_dict.Kinetics.w_decay))

        if not self.cfg.hyper_params.use_I2:
            bounds[self.J + 8] = (0.0, 0.0)
            bounds[self.J + 9] = (0.0, 0.0)

        # Strategy Construction
        strat_cfg = self.cfg.optimization.strategy
        strategy_str = f"{strat_cfg.mutation}{strat_cfg.n_vectors}{strat_cfg.crossover}"

        # Segment Learning Settings
        num_segments = 5
        segment_ratio = 0.3
        
        full_len = len(self.Input_full)
        seg_len = int(full_len * segment_ratio)
        segment_params_list = []
        
        print(f"=== Starting Segment Learning ({num_segments} segments) ===")
        print(f"Saving to: {self.base_save_path}")
        
        with mlflow.start_run():
            mlflow.log_params(self.cfg.hyper_params)
            mlflow.log_param("optimization_strategy", "Segment_Median")

            # --- 1. Loop Segments ---
            for i in range(num_segments):
                start_idx = np.random.randint(0, full_len - seg_len)
                end_idx = start_idx + seg_len
                
                self.current_input = self.Input_full[start_idx:end_idx]
                self.current_output = self.Output_full[start_idx:end_idx]
                
                print(f"\n--- Segment {i+1}/{num_segments} (Range: {start_idx}-{end_idx}) ---")
                
                # Optimization
                result = differential_evolution(
                    self.lnk_model, 
                    bounds,
                    maxiter=self.cfg.optimization.maxiter // 2,
                    popsize=15,
                    updating=self.cfg.optimization.updating,
                    workers=self.cfg.optimization.workers,
                    strategy=strategy_str,
                    disp=True,
                    callback=self.callback
                )
                segment_params_list.append(result.x)
                mlflow.log_metric(f"segment_{i}_score", result.fun)
                
                # Save Segment Data
                # Folder: scripts/segments/{data_name}/{i+1}_{timestamp}/
                seg_folder_name = f"{i+1}_{self.timestamp}"
                seg_save_path = self.base_save_path / seg_folder_name
                
                print(f"Saving segment result to: {seg_save_path}")
                self._simulate_and_save(
                    result.x, 
                    self.current_input, 
                    self.current_output, 
                    seg_save_path, 
                    label=f"Segment_{i+1}"
                )

            # --- 2. Median & Final Verification ---
            print("\n=== Calculating Median Parameters ===")
            segment_params_list = np.array(segment_params_list)
            median_params = np.median(segment_params_list, axis=0)
            
            print("\n=== Final Verification on Full Data ===")
            self.current_input = self.Input_full
            self.current_output = self.Output_full
            
            # Initial Check
            median_score = self.lnk_model(median_params)
            print(f"Score with Median Params: {median_score}")
            mlflow.log_metric("median_params_score", median_score)
            
            # Refinement
            print("Refining with Powell method...")
            final_res = minimize(
                self.lnk_model, 
                median_params, 
                method='Powell', 
                bounds=bounds,
                options={'disp': True, 'maxiter': 200}
            )
            
            best_params = final_res.x
            final_score = final_res.fun
            print(f"Final Refined Score: {final_score}")
            mlflow.log_metric("final_score", final_score)
            
            # Save Final Results
            # Folder: scripts/segments/{data_name}/{timestamp}/
            final_folder_name = f"{self.timestamp}"
            final_save_path = self.base_save_path / final_folder_name
            
            print(f"Saving final result to: {final_save_path}")
            
            # Save Predictions & Plots
            self._simulate_and_save(
                best_params, 
                self.Input_full, 
                self.Output_full, 
                final_save_path, 
                label="Final_Refined"
            )
            
            # Save Config
            with open(final_save_path / "config.yaml", "w") as f:
                OmegaConf.save(self.cfg, f)
            
            # MLflow Artifacts (Optional: copy files to MLflow)
            mlflow.log_artifact(str(final_save_path / "params.txt"))
            mlflow.log_artifact(str(final_save_path / "predict.txt"))
            mlflow.log_artifact(str(final_save_path / "plot.png"))
            
            print("Training and Logging Completed.")

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    model = SegmentLearningModel(cfg)
    model.run()

if __name__ == "__main__":
    main()