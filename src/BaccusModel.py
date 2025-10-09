# BaccusModel.py
# -*- coding: utf-8 -*-
import os
import time
import pprint
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import differential_evolution
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import mlflow
import components.F_LNK as F_LNK
import components.N_LNK as N_LNK
import components.K_baccus as K_LNK

def save_results(data, filepath):
    """
    結果を指定されたファイルパスに保存します。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if isinstance(data, np.ndarray):
        np.savetxt(filepath, data, fmt='%.6f')
    else:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(str(data) + '\n')

### MLflow ###
def flatten_dict_config(cfg: DictConfig) -> dict:
    """
    HydraのネストしたDictConfigをフラットなdictに変換します。
    """
    # OmegaConf.to_containerを使用して、DictConfigをPythonのdictに変換
    d = OmegaConf.to_container(cfg, resolve=True)
    
    # dictをフラット化
    flat_d = {}
    def _flatten(obj, prefix=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f'{prefix}{k}.')
        elif isinstance(obj, list):
            flat_d[prefix[:-1]] = str(obj)
        else:
            flat_d[prefix[:-1]] = obj
    _flatten(d)
    return flat_d


class BaccusOptimizer:
    """
    Hydraの設定を使用してBaccusモデルの最適化を管理するクラス。
    """
    def __init__(self, cfg: DictConfig):
        """
        コンストラクタ
        """
        self.cfg = cfg
        self.total_lnk_model_runs = 0
        self.failed_lnk_model_runs = 0
        self.current_epoch_best_fun_value = 1000.0  # 最小化問題なので初期値は大きな値
        self.epoch_counter = 0
        self.date_str = time.strftime("%Y%m%d_%H")
        
        input_path = to_absolute_path(cfg.data.input_file)
        output_path = to_absolute_path(cfg.data.output_file)
        
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"最適化開始時間: {start_time}\n")
        print(f"データセット '{self.cfg.data.name}' を使用します。")
        print(f"入力データ: {input_path}")
        print(f"出力データ: {output_path}")

        self.Input = np.genfromtxt(input_path)
        self.Output = np.genfromtxt(output_path)
        self.J = self.cfg.hyper_params.J
        
        base_dir = get_original_cwd()
        # self.results_dir のパスを修正し、重複した行を削除
        self.results_dir = os.path.join(base_dir, 'scripts', 'results', f'Baccus_{self.cfg.data.name}', self.date_str)
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"\n結果ファイルは {self.results_dir} に保存されます。")

    def lnk_model(self, x, save_states=False):
        """
        目的関数。与えられたパラメータxでモデルを評価します。
        """
        self.total_lnk_model_runs += 1
        try:
            hp = self.cfg.hyper_params
            dt, R_start, A_start, I1_start, I2_start, tau = \
                hp.dt, hp.R_start, hp.A_start, hp.I1_start, hp.I2_start, hp.tau

            J = self.J
            alphas = x[0:J]
            delta = x[J]
            a_nonlinear = x[J+1]
            b1_nonlinear = x[J+2]
            b2_nonlinear = x[J+3]
            ka_kinetic = x[J+4]
            kfi_kinetic = x[J+5]
            kfr_kinetic = x[J+6]
            ksi_kinetic = x[J+7]
            ksr_kinetic = x[J+8]

            t = min(len(self.Input), len(self.Output))
            
            # 1. Linear Filter
            linear_filter_kernel, _ = F_LNK.main(alphas, delta, t, dt, tau)
            g_t = np.convolve(self.Input[:t], linear_filter_kernel, mode='same')

            # 2. Nonlinear Model
            u_t = N_LNK.main(g_t, a_nonlinear, b1_nonlinear, b2_nonlinear)

            # 3. Kinetic Model
            R_state, A_state, I1_state, I2_state, check = K_LNK.main(
                len(u_t), u_t, dt, R_start, A_start, I1_start, I2_start,
                ka_kinetic, kfi_kinetic, kfr_kinetic, ksi_kinetic, ksr_kinetic,
                label=f"LNK_run {self.total_lnk_model_runs}"
            )
            # print(f"Check status for LNK model run {self.total_lnk_model_runs}: {check}", end='\r', flush=True)
            if check == 1:
                print(f"Check status for LNK model run {self.total_lnk_model_runs}: {check}", end='\r', flush=True)
            else:
                print(f"Check status for LNK model run {self.total_lnk_model_runs}: {check}", end='\r', flush=True)

            # 4. Evaluation
            correlation = 1.0  # ペナルティ値
            if check == 1:
                keep_post = A_state[:t]
                output_trimmed = self.Output[:t]
                correlation, _ = spearmanr(output_trimmed, keep_post)
                correlation = -1 * correlation  # 最小化のため
            else:
                self.failed_lnk_model_runs += 1

            self.current_epoch_best_fun_value = correlation

            if save_states:
                return correlation, R_state, A_state, I1_state, I2_state
            else:
                return correlation
        except Exception as e:
            print(f"エラー内容: {e}")
            import traceback
            traceback.print_exc() # 詳細なエラー情報を表示
            
            self.failed_lnk_model_runs += 1
            return 1.0  # エラー時は大きなペナルティを返す

    def save_intermediate_results(self, xk, convergence=None):
        """
        各エポックの終わりに呼び出されるコールバック関数。
        """
        total_run = self.total_lnk_model_runs
        self.epoch_counter += 1
        current_best_correlation_value = -self.lnk_model(xk, save_states=False) 
        intermediate_dir = os.path.join(self.results_dir, 'epochs')
        os.makedirs(intermediate_dir, exist_ok=True)
        save_results(xk, os.path.join(intermediate_dir, f'epoch_{self.epoch_counter:03d}_params.txt'))
        save_results(current_best_correlation_value, os.path.join(intermediate_dir, f'epoch_{self.epoch_counter:03d}_correlation.txt'))
        
        # 各エポックごとの全パラメータの値をメトリクスとして記録する
        
        intermediate_params = {
            **{f'L{i+1}': xk[i] for i in range(self.J)},
            'delta': xk[self.J], 'a': xk[self.J+1], 'b1': xk[self.J+2], 'b2': xk[self.J+3],
            'ka': xk[self.J+4], 'kfi': xk[self.J+5], 'kfr': xk[self.J+6], 'ksi': xk[self.J+7], 'ksr': xk[self.J+8]
        }
        
        # mlflow.log_metrics を使って辞書の中身を一度に記録
        # keyの先頭に "epoch_" をつけて、最終結果(optimal_)と区別する
        metrics_to_log = {f"epoch_{k}": v for k, v in intermediate_params.items()}
        metrics_to_log["epoch_correlation"] = current_best_correlation_value
        mlflow.log_metrics(metrics_to_log, step=self.epoch_counter)

        timestamp = time.strftime("%d_%H%M%S")
        tqdm.write(
            # 表示する値も再計算したものを使用する
            f"---{timestamp} | Epoch {self.epoch_counter:03d} Saved | Correlation: {current_best_correlation_value:.4f} | Total Runs: {total_run} ---"
        )
    def save_optimal_results(self, optimal_params, optimal_correlation, R_state, A_state, I1_state, I2_state):
        """
        最終的な最適化結果を保存します。
        """
        print(f"\n最適化結果を {self.results_dir} に保存中...")

        param_map = {
            **{f'L{i+1}': optimal_params[i] for i in range(self.J)},
            'delta': optimal_params[self.J],
            'a': optimal_params[self.J+1],
            'b1': optimal_params[self.J+2],
            'b2': optimal_params[self.J+3],
            'ka': optimal_params[self.J+4],
            'kfi': optimal_params[self.J+5],
            'kfr': optimal_params[self.J+6],
            'ksi': optimal_params[self.J+7],
            'ksr': optimal_params[self.J+8],
            'correlation': optimal_correlation
        }

        # 最終的なパラメータをファイルに保存
        for name, val in param_map.items():
            save_results(val, os.path.join(self.results_dir, f'{name}.txt'))
        save_results(optimal_correlation, os.path.join(self.results_dir, 'correlation.txt'))

        # 最終的な値をMLflowのMetricsとして記録
        final_metrics = {f"optimal_{k}": v for k, v in param_map.items()}
        final_metrics["optimal_correlation"] = optimal_correlation
        mlflow.log_metrics(final_metrics)

        state_dir = os.path.join(self.results_dir, 'state')
        os.makedirs(state_dir, exist_ok=True)
        save_results(R_state, os.path.join(state_dir, 'R_state.txt'))
        save_results(A_state, os.path.join(state_dir, 'A_state.txt'))
        save_results(I1_state, os.path.join(state_dir, 'I1_state.txt'))
        save_results(I2_state, os.path.join(state_dir, 'I2_state.txt'))
        
        print("保存が完了しました。")

    def run(self):
        """
        最適化プロセスを実行します。
        """
        try_bounds = [
            (-1.0, 2.0) for _ in range(self.J)  # alphas (L1-L15)
        ] + [
            (0.05, 1.0),   # delta
            (0.1, 10.0),   # a (nonlinear)
            (0.0, 5.0),    # b1 (nonlinear)
            (-1.0, 0.0),   # b2 (nonlinear)
            (-0.01, 2.0),  # ka (kinetic)
            (0.5, 5.0),    # kfi (kinetic)
            (-0.01, 0.3),  # kfr (kinetic)
            (0.001, 0.1),  # ksi (kinetic)
            (0.001, 0.1)   # ksr (kinetic)
        ]
        
        print(f"Number of parameters to optimize: {len(try_bounds)}")
        print("差分進化法による最適化を開始します...")
        
        opt_cfg = self.cfg.optimization
        result = differential_evolution(
            self.lnk_model, 
            try_bounds, 
            disp=True,
            updating=opt_cfg.updating, 
            maxiter=opt_cfg.maxiter, 
            popsize=opt_cfg.popsize, 
            strategy=opt_cfg.strategy, 
            workers=opt_cfg.workers, 
            callback=self.save_intermediate_results
        )

        print("\n最適化が完了しました。")
        pprint.pprint(result)

        optimal_params = result.x
        optimal_correlation = -result.fun

        print("\n最終的な状態を取得するため、最適なパラメータでモデルを再実行します...")
        _, r_final, a_final, i1_final, i2_final = self.lnk_model(optimal_params, save_states=True)

        if a_final is not None:
            self.save_optimal_results(optimal_params, optimal_correlation, r_final, a_final, i1_final, i2_final)
            mlflow.log_artifacts(self.results_dir, artifact_path="results")
        else:
            print("Kineticモデルが最終実行で失敗したため、状態は保存されません。")
            print(f"最終的な相関係数: {optimal_correlation:.4f}")
            # 失敗した場合でも、最終的な相関係数だけは記録しておく
            mlflow.log_metric("final_correlation_on_failure", optimal_correlation)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Hydraによって呼び出されるメイン関数。
    """
    original_cwd = get_original_cwd()
    mlruns_path = os.path.join(original_cwd, 'scripts', 'mlruns')
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    
    # 1. Experiment（実験）を設定。同じ名前の実験はグループ化される
    mlflow.set_experiment(f"Baccus_Optimization_{cfg.data.name}")

    # 2. Run（実行）を開始。with文を使うと、ブロックを抜ける際に自動で終了処理が行われる
    # run_nameで、UIに表示される実行の名前を設定
    run_name = f"{cfg.optimization.strategy}_{time.strftime('%Y%m%d_%H')}"
    with mlflow.start_run(run_name=run_name):
        flat_params = flatten_dict_config(cfg)# 3. Hydraの設定（ハイパーパラメータ）をMLflowに記録
        mlflow.log_params(flat_params) # ネストした設定ファイルが見やすいようにフラット化する
        
        # 4. タグを設定して、後で検索やフィルタリングをしやすくする
        mlflow.set_tag("data_name", cfg.data.name)
        mlflow.set_tag("optimizer", "differential_evolution")

        # 5. 最適化プロセスを実行
        optimizer = BaccusOptimizer(cfg)
        optimizer.run()

if __name__ == "__main__":
    main()