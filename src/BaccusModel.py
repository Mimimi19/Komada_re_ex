# BaccusModel.py
# -*- coding: utf-8 -*-
import os
import time
import pprint
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import mlflow
import requests
from dotenv import load_dotenv
import components.L_LNK as L_LNK
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

import datetime
try:
    import zoneinfo
except ImportError:
    # Python 3.8など古い環境用のフォールバック
    from backports import zoneinfo

class BaccusOptimizer:
    """
    Hydraの設定を使用してBaccusモデルの最適化を管理するクラス。
    """
    def __init__(self, cfg: DictConfig):
        """
        コンストラクタ
        """
        
        self.cfg = cfg
        # I2を使用するかどうかのフラグ
        self.use_I2 = self.cfg.hyper_params.get('use_I2', True)
        
        self.total_lnk_model_runs = 0
        self.failed_lnk_model_runs = 0
        self.current_epoch_best_fun_value = 1000.0  # 最小化問題なので初期値は大きな値
        self.epoch_counter = 0
        
        input_path = to_absolute_path(cfg.data.input_file)
        output_path = to_absolute_path(cfg.data.output_file)
        
        # タイムゾーンを'Asia/Tokyo'に指定して現在時刻を取得
        jst = zoneinfo.ZoneInfo("Asia/Tokyo")
        start_time = datetime.datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S")
        self.date_str = time.strftime("%Y%m%d_%H")
        
        #debug用ログファイルの準備
        log_path = os.path.join(get_original_cwd(), "scripts", "lnk_model_debug.log")
        self.debug_log_path = log_path
        os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
        with open(self.debug_log_path, "w") as f:
            f.write("LNK Model Debug Log\n")
            
        print(f"最適化開始時間: {start_time}\n")
        print(f"データセット '{self.cfg.data.name}' を使用します。")
        print(f"入力データ: {input_path}")
        print(f"出力データ: {output_path}")
        
        if not self.use_I2:
            print("\n--- 警告 ---")
            print("hyper_params.use_I2 が False に設定されています。")
            print("I2 state (p_I2, ksi, ksr) は強制的に 0.0 として扱われます。")
            print("------------\n")
        else:
            print("\nI2 state (p_I2, ksi, ksr) は最適化対象に含まれます。\n")

        # 1. 入力データの正規化 (Z-score Standardization)
        # 光刺激は「平均からの変化量(コントラスト)」として扱うのがモデルにとって最適です。
        raw_input = np.genfromtxt(input_path)
        input_std = np.std(raw_input)
        if input_std > 1e-9: # ゼロ除算防止
            # 平均を0、標準偏差を1にする（これでどんな単位のデータが来てもモデルへの入力は -2.0 ~ +2.0 程度に収まります）
            Input_full = (raw_input - np.mean(raw_input)) / input_std
        else:
            # 変化がないデータの場合（エラー回避）
            Input_full = raw_input - np.mean(raw_input)

        # 2. 出力データの正規化 (Max-Abs Scaling)
        # カルシウム応答などの生体信号は単位が実験ごとに違うため、最大値が 1.0 になるように揃えます。
        raw_output = np.genfromtxt(output_path)
        max_val = np.max(np.abs(raw_output))
        if max_val > 1e-9: # ゼロ除算防止
            Output_full = raw_output / max_val
        else:
            Output_full = raw_output
                
        self.J = self.cfg.hyper_params.J
        
        # データの最初と最後をトリミング
        try:
            dt = self.cfg.data.dt  # サンプリング間隔 (秒)
            trim_i_seconds = self.cfg.hyper_params.trim_I_seconds # 入力データのトリミング秒数
            trim_o_seconds = self.cfg.hyper_params.trim_O_seconds # 出力データのトリミング秒数
            # トリミングするインデックス数を計算
            trim_i_indices = int(trim_i_seconds / dt)
            trim_o_indices = int(trim_o_seconds / dt)
            
            min_len = min(len(Input_full), len(Output_full))
            
            # 配列がトリミング分（合計）より短い場合の安全チェック
            if min_len <= (trim_i_indices + trim_o_indices):
                print(f"警告: データ長 ({min_len}) が短すぎて、前後{trim_i_seconds+trim_o_seconds}秒（{trim_i_indices+trim_o_indices}インデックス）をトリミングできません。")
                print("トリミングせずに処理を続行します。")
                self.Input = Input_full
                self.Output = Output_full
            else:
                start_index = trim_i_indices
                
                if trim_o_indices > 0:
                    end_index = -trim_o_indices 
                else:
                    end_index = len(Input_full) # スライスで[start:None]と同じ意味
                
                self.Input = Input_full[start_index:end_index]
                self.Output = Output_full[start_index:end_index]
                
                print(f"\n--- データトリミング (前{trim_i_seconds}秒, 後{trim_o_seconds}秒) ---")
                print(f"dt={dt}s のため、前{trim_i_indices}インデックス、後{trim_o_indices}インデックスをトリミングします。")
                
                # 表示用の終端インデックスを計算
                end_idx_pos_in = len(Input_full) - trim_o_indices - 1
                end_idx_pos_out = len(Output_full) - trim_o_indices - 1
                
                print(f"Input: {len(Input_full)} -> {len(self.Input)} (インデックス {start_index} から {end_idx_pos_in} を使用)")
                print(f"Output: {len(Output_full)} -> {len(self.Output)} (インデックス {start_index} から {end_idx_pos_out} を使用)")
                print(f"----------------------------------\n")
                
        except Exception as e:
            print(f"エラー: データトリミング中に失敗しました。{e}")
            print("トリミングせずに処理を続行します。")
            self.Input = Input_full
            self.Output = Output_full
        
        base_dir = get_original_cwd()
        # self.results_dir のパスを修正し、重複した行を削除
        self.results_dir = os.path.join(base_dir, 'scripts', 'results', f'Baccus_{self.cfg.data.name}', self.date_str)
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"\n結果ファイルは {self.results_dir} に保存されます。")
        
    def _normalize_states(self, p_R, p_A, p_I1, p_I2):
        """
        初期占有率の比率を正規化し、合計1.0のタプルを返すヘルパー関数。
        """
        # use_I2: false の場合、p_I2 は 0.0 が渡される想定
        total = p_R + p_A + p_I1 + p_I2
        
        if total > 1e-9: # ゼロ除算を回避
            R_start = p_R / total
            A_start = p_A / total
            I1_start = p_I1 / total
            I2_start = p_I2 / total
            return R_start, A_start, I1_start, I2_start
        else:
            # オプティマイザが全て0を提案した場合のフォールバック
            # (この試行はペナルティ(相関1.0)を受ける)
            return 1.0, 0.0, 0.0, 0.0

    def lnk_model(self, x, save_states=False):
        """
        目的関数。与えられたパラメータxでモデルを評価します。
        mode='full' に変更して因果性を明確にし、初期の余分な部分をカットします。
        """
        self.total_lnk_model_runs += 1
        try:
            hp = self.cfg.hyper_params
            dt = self.cfg.data.dt
            tau = hp.tau
            
            J = self.J
            
            alphas = x[0:J]
            delta = x[J]
            a_nonlinear = x[J+1]
            kappa_nonlinear = x[J+2]
            b1_nonlinear = x[J+3]
            b2_nonlinear = x[J+4]
            ka_kinetic = x[J+5]
            kfi_kinetic = x[J+6]
            kfr_kinetic = x[J+7]
            ksi_kinetic = x[J+8]
            ksr_kinetic = x[J+9]
            # 各占有率の初期値
            p_R = x[J+10]
            p_A = x[J+11]
            p_I1 = x[J+12]
            p_I2 = x[J+13]
            
            if not self.use_I2:
                ksi_kinetic = 0.0
                ksr_kinetic = 0.0
                p_I2 = 0.0

            R_start, A_start, I1_start, I2_start = self._normalize_states(
                p_R, p_A, p_I1, p_I2
            )
            
            # フィルタの有効長 (tau) に必要なポイント数だけを計算します。
            filter_points = int(tau / dt) + 1  # +1 は余裕を持たせるため
            
            # カーネル生成 (長さは filter_points のみ)
            linear_filter_kernel, _ = L_LNK.main(alphas, delta, filter_points, dt, tau)
            
            #畳み込みとサイズ調整
            g_full = fftconvolve(self.Input, linear_filter_kernel, mode='full')
            
            # フィルタによる位相遅れを補正するためのシフト量
            # Kernelは L_LNK 内で反転([::-1])されているため、ピーク位置等を考慮して調整します
            # ここでは単純にフィルタ長分をシフトして「因果的」に合わせます
            shift_idx = int(tau / dt) 
            
            # データ長に合わせて切り出し
            if len(g_full) > shift_idx + len(self.Input):
                g_t = g_full[shift_idx : shift_idx + len(self.Input)]
            else:
                # 万が一長さが足りない場合のフォールバック（後ろをゼロ埋めなど）
                g_t = g_full[:len(self.Input)]

            # これがないと g_t が ±100 になり、非線形関数が飽和する
            g_std = np.std(g_t)
            if g_std > 1e-9:
                g_t = g_t / g_std
                
            # Nonlinear Model
            u_t = N_LNK.main(g_t, a_nonlinear, kappa_nonlinear, b1_nonlinear, b2_nonlinear, ka_kinetic)
            # 飽和ペナルティ 
            # u_t の標準偏差が極端に小さい（平坦）、または値が張り付いている場合にペナルティ
            # ここでは簡単に「u_tの分散が小さすぎる場合」を検知
            if np.std(u_t) < 1e-6:  # 閾値は調整が必要
                print("Penalty: Saturation detected", end='\r', flush=True)
                return 1.0 # 悪いスコア（相関1.0相当のペナルティ）として返す
            # Kinetic Model
            R_state, A_state, I1_state, I2_state, check = K_LNK.main(
                len(u_t), u_t, dt, R_start, A_start, I1_start, I2_start,
                ka_kinetic, kfi_kinetic, kfr_kinetic, ksi_kinetic, ksr_kinetic,
                label=f"LNK_run {self.total_lnk_model_runs}"
            )
            
            with open(self.debug_log_path, "a") as f:
                f.write(f"Run: {self.total_lnk_model_runs}, Check: {check}\n")
                
            print(f"Check status for LNK model run {self.total_lnk_model_runs}: {check}", end='\r', flush=True)

            # Evaluation
            correlation = 1.0  # ペナルティ値
            if check == 1:
                # 1. まずデータ長を合わせる
                current_len = len(A_state)
                output_aligned = self.Output[:current_len]
                model_aligned = A_state  # A_stateはマスク前の生データを使う

                # 2. 万が一長さが合わない場合の安全策
                if len(output_aligned) != len(model_aligned):
                     min_l = min(len(output_aligned), len(model_aligned))
                     output_aligned = output_aligned[:min_l]
                     model_aligned = model_aligned[:min_l]

                # 3. マスク処理: 最初の 1秒スライスして捨てる、Aの初期値依存部分を評価から除外
                mask_seconds = 1.0  # または 2.0
                mask_idx = int(mask_seconds / dt)

                if mask_idx < len(output_aligned):
                    # スライスしたデータ同士で相関を計算
                    # これにより初期値依存の不安定な部分を完全に無視できる
                    output_eval = output_aligned[mask_idx:]
                    model_eval = model_aligned[mask_idx:]
                    
                    # 標準偏差チェック（平坦な線になっていないか）
                    if np.std(model_eval) > 1e-9 and np.std(output_eval) > 1e-9:
                        corr_val, _ = spearmanr(output_eval, model_eval)
                        correlation = -1 * corr_val
                    else:
                        correlation = 0.0 # 平坦ならスコアなし
                else:
                    correlation = 1.0 
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
            traceback.print_exc()
            self.failed_lnk_model_runs += 1
            return 1.0  # ペナルティ値
        
    def save_intermediate_results(self, xk, convergence=None):
        """
        各エポックの終わりに呼び出されるコールバック関数。
        """
        
        self.epoch_counter += 1
        current_best_correlation_value = -self.lnk_model(xk, save_states=False) 
        intermediate_dir = os.path.join(self.results_dir, 'epochs')
        os.makedirs(intermediate_dir, exist_ok=True)
        save_results(xk, os.path.join(intermediate_dir, f'epoch_{self.epoch_counter:03d}_params.txt'))
        save_results(current_best_correlation_value, os.path.join(intermediate_dir, f'epoch_{self.epoch_counter:03d}_correlation.txt'))
        
        # 各エポックごとの全パラメータの値をメトリクスとして記録する
        
        intermediate_params = {
            **{f'L{i+1}': xk[i] for i in range(self.J)},
            'delta': xk[self.J], 'a': xk[self.J+1], 'kappa': xk[self.J+2], 'b1': xk[self.J+3], 'b2': xk[self.J+4],
            'ka': xk[self.J+5], 'kfi': xk[self.J+6], 'kfr': xk[self.J+7], 
            'ksi': xk[self.J+8], 'ksr': xk[self.J+9],
            # 占有率
            'p_R': xk[self.J+10],
            'p_A': xk[self.J+11],
            'p_I1': xk[self.J+12],
            'p_I2': xk[self.J+13]
        }
        
        # mlflow.log_metrics を使って辞書の中身を一度に記録
        # keyの先頭に "epoch_" をつけて、最終結果(optimal_)と区別する
        metrics_to_log = {f"epoch_{k}": v for k, v in intermediate_params.items()}
        metrics_to_log["epoch_correlation"] = current_best_correlation_value
        mlflow.log_metrics(metrics_to_log, step=self.epoch_counter)

        timestamp = time.strftime("%d_%H:%M:%S")
        tqdm.write(
            # 表示する値も再計算したものを使用する
            f"---{timestamp} | Epoch {self.epoch_counter:03d} Saved | Correlation: {current_best_correlation_value:.4f} ---"
        )
    def save_optimal_results(self, optimal_params, optimal_correlation, R_state, A_state, I1_state, I2_state):
        """
        最終的な最適化結果を保存します。
        """
        print(f"\n最適化結果を {self.results_dir} に保存中...")

        try:
            param_map = {}
            
            # --- 配列から辞書へのマッピング (インデックス管理を安全に) ---
            # lnk_model内の展開順序と厳密に一致させます
            idx = 0
            
            # Linear Filter (L1...LJ)
            for i in range(self.J):
                param_map[f'L{i+1}'] = optimal_params[idx]
                idx += 1
            
            # Scalar Parameters
            # 配列のインデックスを順番に進めていくため、記述ミスによるズレを防げます
            param_keys = [
                'delta', 'a', 'kappa', 'b1', 'b2', 
                'ka', 'kfi', 'kfr', 'ksi', 'ksr',
                'p_R', 'p_A', 'p_I1', 'p_I2'
            ]
            
            for key in param_keys:
                param_map[key] = optimal_params[idx]
                idx += 1
                
            # 相関係数もマップに追加
            param_map['correlation'] = optimal_correlation

            # --- 正規化された初期状態の計算 ---
            # use_I2フラグを考慮して正規化計算を行う
            p_R_opt = param_map['p_R']
            p_A_opt = param_map['p_A']
            p_I1_opt = param_map['p_I1']
            p_I2_opt = param_map['p_I2'] if self.use_I2 else 0.0

            R_start_opt, A_start_opt, I1_start_opt, I2_start_opt = self._normalize_states(
                p_R_opt, p_A_opt, p_I1_opt, p_I2_opt
            )

            param_map['R_start_normalized'] = R_start_opt
            param_map['A_start_normalized'] = A_start_opt
            param_map['I1_start_normalized'] = I1_start_opt
            param_map['I2_start_normalized'] = I2_start_opt

            # ---ファイル保存ループ (エラーハンドリング付き) ---
            saved_count = 0
            for name, val in param_map.items():
                try:
                    # 個別のファイル保存に失敗しても他は保存し続ける
                    save_results(val, os.path.join(self.results_dir, f'{name}.txt'))
                    saved_count += 1
                except Exception as e:
                    print(f"警告: {name}.txt の保存に失敗しました: {e}")

            print(f"パラメータ保存完了: {saved_count}/{len(param_map)} ファイル")

            # --- MLflowへの記録 ---
            try:
                final_metrics = {f"optimal_{k}": v for k, v in param_map.items()}
                mlflow.log_metrics(final_metrics)
            except Exception as e:
                print(f"警告: MLflowへのメトリクス送信に失敗しました: {e}")

            # --- 時系列状態変数の保存 ---
            state_dir = os.path.join(self.results_dir, 'state')
            os.makedirs(state_dir, exist_ok=True)
            
            save_results(R_state, os.path.join(state_dir, 'R_state.txt'))
            save_results(A_state, os.path.join(state_dir, 'A_state.txt'))
            save_results(I1_state, os.path.join(state_dir, 'I1_state.txt'))
            save_results(I2_state, os.path.join(state_dir, 'I2_state.txt'))
            
            print("すべての保存処理が完了しました。")
            
        except Exception as e:
            print(f"重大なエラー: save_optimal_results 内で予期せぬエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            
    def run(self):
        """
        最適化プロセスを実行します。
        """
        # ワークステーションでの並列処理の際にNumbaのJITが渋滞する問題を回避するためのウォームアップ
        print("Numba JITコンパイラのウォームアップ中...")
        try:
            # ダミーのパラメータ配列 (長さ: J + 14) を作成
            x_dummy = np.ones(self.J + 14) 
            # 目的関数を一度だけ実行して、コンパイルを強制する
            self.lnk_model(x_dummy, save_states=False)
            print("ウォームアップ完了。最適化を開始します。")
        except Exception as e:
            print(f"警告: ウォームアップ中にエラーが発生しました: {e}")
            # エラーが起きても、本番の最適化は続行してみる
            
        # Configからパラメータ境界(param_bounds)を取得
        pb = self.cfg.hyper_params.param_bounds
        J = self.J
        
        # alphas (L1-L15) の境界を生成
        alpha_bounds_tuple = tuple(pb.LinearFilter.alphas)
        try_bounds = [alpha_bounds_tuple] * J
        
        #    Configのリスト [min, max] を tuple(min, max) に変換
        try:
            try_bounds.extend([
                # LinearFilter
                tuple(pb.LinearFilter.delta),
                # Nonlinear
                tuple(pb.Nonlinear.a),
                tuple(pb.Nonlinear.kappa),
                tuple(pb.Nonlinear.b1),
                tuple(pb.Nonlinear.b2),
                # Kinetics
                tuple(pb.Kinetics.ka),
                tuple(pb.Kinetics.kfi),
                tuple(pb.Kinetics.kfr),
                tuple(pb.Kinetics.ksi),
                tuple(pb.Kinetics.ksr),
            # InitialStates 
                tuple(pb.InitialStates.p_R),
                tuple(pb.InitialStates.p_A),
                tuple(pb.InitialStates.p_I1),
                tuple(pb.InitialStates.p_I2)
            ])
        except Exception as e:
            print(f"エラー: config.yaml の 'param_bounds' 設定が不足しています。")
            print("param_bounds に 'InitialStates' (p_R, p_A, p_I1, p_I2) が正しく定義されているか確認してください。")
            print(f"詳細: {e}")
            raise

        
        #use_I2=False の場合、探索範囲を [0, 0] に固定
        if not self.use_I2:
            print("I2が無効なため、ksi, ksr, p_I2 の探索範囲を [0.0, 0.0] に固定します。")
            ksi_index = J + 8
            ksr_index = J + 9
            p_I2_index = J + 13 # p_I2 のインデックス
            
            try_bounds[ksi_index] = (0.0, 0.0)
            try_bounds[ksr_index] = (0.0, 0.0)
            try_bounds[p_I2_index] = (0.0, 0.0) # I2の比率も0に固定
        # パラメータ探索範囲(try_bounds)をMLflowに記録する
        param_names = [f'L{i+1}' for i in range(self.J)] + [
            'delta', 'a', 'kappa', 'b1', 'b2', 'ka', 'kfi', 'kfr', 'ksi', 'ksr',
            'p_R', 'p_A', 'p_I1', 'p_I2'
        ]

        #MLflowに記録するための辞書を作成
        bounds_to_log = {}
        for name, bound_tuple in zip(param_names, try_bounds):
            # (下限, 上限) のタプルを文字列に変換して辞書に追加
            # キーの先頭に 'bound_' をつけて、他のパラメータと区別する
            bounds_to_log[f"bound_{name}"] = str(bound_tuple)
        
        # 作成した辞書をに記録
        mlflow.log_params(bounds_to_log)
        
        print(f"Number of parameters to optimize: {len(try_bounds)}")
        print("差分進化法による最適化を開始します...")
        
        opt_cfg = self.cfg.optimization
        # Hydra設定からstrategyコンポーネントを取得
        try:
            strategy_cfg = opt_cfg.strategy
            mutation = strategy_cfg.mutation
            n_vectors = strategy_cfg.n_vectors
            crossover = strategy_cfg.crossover
            # scipy.optimize.differential_evolution が要求する strategy 文字列を組み立てる
            de_strategy_str = f"{mutation}{n_vectors}{crossover}"
            
            # 組み立てた戦略をログに出力
            print(f"DE戦略: {mutation}/{n_vectors}/{crossover} (scipy strategy: '{de_strategy_str}')")
            with open(self.debug_log_path, "a") as f:
                    f.write(f"DE戦略: {mutation}/{n_vectors}/{crossover} (scipy strategy: '{de_strategy_str}')\n")
                    
        except Exception as e:
            print(f"エラー: 'optimization.strategy' の設定が不正です。")
            print("config.yaml で mutation, n_vectors, crossover が正しく設定されているか確認してください。")
            print(f"詳細: {e}")
            raise # エラーが発生したら最適化を実行せずに終了
        
        de_result = differential_evolution(
            self.lnk_model,      # 目的関数（最小化したい関数）
            try_bounds,          # 探索するパラメータの範囲（各変数の上下限）
            disp=True,           # 途中経過を表示する
            updating=opt_cfg.updating,   # 並列計算や更新方法の設定
            maxiter=opt_cfg.maxiter,     # 最大繰り返し回数
            popsize=opt_cfg.popsize,     # 個体数（探索候補の数）
            strategy=de_strategy_str,  # 差分進化の戦略（mutation の方法）
            workers=opt_cfg.workers,     # 並列実行のためのスレッド・プロセス数
            callback=self.save_intermediate_results  # 各イテレーション後に呼ばれる関数
        )

        print("\n大域探索 (DE) が完了しました。")
        print(f"DE 最良スコア: {-de_result.fun:.6f}")
        print("最適な結果を初期値として局所探索 (Powell) を開始します... (Phase 2: Refinement)")

        #局所探索 (Powell法で解を「磨き上げ」)
        # config.yaml に local_maxiter を追加するか、ここではDEのイテレーション数を流用
        local_maxiter = opt_cfg.get('local_maxiter', opt_cfg.maxiter // 2) 

        result = minimize(
            self.lnk_model,          # 目的関数
            de_result.x,             # DEで見つけた最適解を初期値 (x0) に設定
            method='Powell',         # 微分不要で高速な局所探索手法
            bounds=try_bounds,       # 境界制約はそのまま維持
            options={
                'disp': True,        # 局所探索の経過も表示
                'maxiter': local_maxiter # 局所探索用のイテレーション数
            }
        )

        print("\nハイブリッド最適化が完了しました。")
        pprint.pprint(result)
        
        print("\n--- 検証統計 ---")
        # 成功した実行回数を計算
        successful_runs = self.total_lnk_model_runs - self.failed_lnk_model_runs
        
        # 成功率を計算 (ゼロ除算を回避)
        if self.total_lnk_model_runs > 0:
            success_rate = (successful_runs / self.total_lnk_model_runs) * 100.0
        else:
            success_rate = 0.0  # 実行がなかった場合

        # ターミナルに表示
        print(f"総試行回数 (total_lnk_model_runs): {self.total_lnk_model_runs}")
        print(f"成功回数 (check=1): {successful_runs}")
        print(f"失敗回数 (check=0 or Error): {self.failed_lnk_model_runs}")
        print(f"成功率: {success_rate:.2f}%")
        
        # MLflowにメトリクスとして保存
        mlflow.log_metric("final_total_runs", self.total_lnk_model_runs)
        mlflow.log_metric("final_successful_runs", successful_runs)
        mlflow.log_metric("final_success_rate_percent", success_rate)
        
        print("検証統計をMLflowに保存しました。")
        
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
    # httpsを突破するために CA証明書のパスを定義
    ca_cert_path = os.path.join(original_cwd, 'ca.crt')
    if os.path.exists(ca_cert_path):
        os.environ['REQUESTS_CA_BUNDLE'] = ca_cert_path
        print(f"カスタムCA証明書を読み込みました: {ca_cert_path}")
    else:
        print(f"警告: CA証明書ファイルが見つかりません: {ca_cert_path}")
    # .env ファイルからNASのURLを読み込む
    dotenv_path = os.path.join(original_cwd, '.env')
    load_dotenv(dotenv_path)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    if tracking_uri:
        print(f"MLflowの保存先をNAS ({tracking_uri}) に設定します。")
        mlflow.set_tracking_uri(tracking_uri)
            
    else:
        print(f"警告: .envファイルまたは MLFLOW_TRACKING_URI が見つかりません。")
        print("フォールバック: ローカルの 'scripts/mlruns' を使用します。")
        mlruns_path = os.path.join(original_cwd, 'scripts', 'mlruns')
        mlflow.set_tracking_uri(f"file:{mlruns_path}")
        
    try:
        # Experiment（実験）を設定。
        mlflow.set_experiment(f"Baccus_Optimization_{cfg.data.name}")
        
    except requests.exceptions.SSLError as e:
        print("\n--- SSL接続エラー ---")
        print("MLflowサーバーへの接続に失敗しました。ca.crtが正しいか、VPNが接続されているか確認してください。")
        print(f"詳細: {e}")
        # エラーが発生したらプログラムを終了する
        raise
    except requests.exceptions.ConnectionError as e:
        print("\n--- ネットワーク接続エラー ---")
        print("MLflowサーバーへの接続に失敗しました。VPNが接続されているか、.envのURLが正しいか確認してください。")
        print(f"詳細: {e}")
        raise
    # Hydra設定から戦略コンポーネントを取得
    try:
        strategy_cfg = cfg.optimization.strategy
        mutation = strategy_cfg.mutation
        n_vectors = strategy_cfg.n_vectors
        crossover = strategy_cfg.crossover
        
        # MLflowの実行名(run_name)用に 'rand/1/bin' 形式の文字列を組み立てる
        strategy_str_for_name = f"{mutation}/{n_vectors}/{crossover}"
        
    except Exception as e:
        print(f"警告: 'optimization.strategy' の設定が不正です。run_nameにデフォルト文字列を使用します。")
        print(f"詳細: {e}")
        strategy_str_for_name = "unknown_strategy" # フォールバック

    # 戦略文字列と日付を使って run_name を定義
    run_name = f"{strategy_str_for_name}_{time.strftime('%Y%m%d_%H%M')}"

    # Run（実行）を開始。with文を使うと、ブロックを抜ける際に自動で終了処理が行われる
    # run_nameで、UIに表示される実行の名前を設定
    
    use_I2_str = "I2-True" if cfg.hyper_params.get('use_I2', True) else "I2-False"
    run_name = f"{cfg.optimization.strategy}_{time.strftime('%Y%m%d_%H')}"
    with mlflow.start_run(run_name=run_name):
        flat_params = flatten_dict_config(cfg)# Hydraの設定（ハイパーパラメータ）をMLflowに記録
        mlflow.log_params(flat_params) # ネストした設定ファイルが見やすいようにフラット化する
        
        # タグを設定して、後で検索やフィルタリングをしやすくする
        mlflow.set_tag("data_name", cfg.data.name)
        mlflow.set_tag("optimizer", "hybrid_DE_Powell")
        mlflow.set_tag("use_I2", str(cfg.hyper_params.get('use_I2', True)))
        # 最適化プロセスを実行
        optimizer = BaccusOptimizer(cfg)
        optimizer.run()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()