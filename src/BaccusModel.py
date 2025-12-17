# src/model/BaccusModel.py
# -*- coding: utf-8 -*-
import os
import time
import pprint
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import mlflow
import requests
from dotenv import load_dotenv

# コンポーネントのインポート
import src.components.L_LNK as L_LNK
import src.components.N_LNK as N_LNK
import src.components.K_baccus as K_LNK

# 目的関数モジュールのインポート (新規追加)
import src.components.objectives.spearman as obj_spearman
import src.components.objectives.hybrid as obj_hybrid

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
        
        # 目的関数のタイプを取得 (デフォルトは spearman)
        self.objective_type = self.cfg.hyper_params.get('objective_type', 'spearman')
        
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
        print(f"目的関数: {self.objective_type}")
        
        if not self.use_I2:
            print("--- 警告: use_I2 = False ---")
        else:
            print("\nI2 state (ksi, ksr) は最適化対象に含まれます。\n") 

        # 入力データの正規化
        raw_input = np.genfromtxt(input_path)
        input_std = np.std(raw_input)
        if input_std > 1e-9: # ゼロ除算防止
            # 平均を0、標準偏差を1にする（これでどんな単位のデータが来てもモデルへの入力は -2.0 ~ +2.0 程度に収まる）
            Input_full = (raw_input - np.mean(raw_input)) / input_std
        else:
            # 変化がないデータの場合（エラー回避）
            Input_full = raw_input - np.mean(raw_input)

        # 出力データの正規化 
        # カルシウム応答などの生体信号は単位が実験ごとに違うため、最大値が 1.0 になるように揃える
        raw_output = np.genfromtxt(output_path)
        max_val = np.max(np.abs(raw_output))
        if max_val > 1e-9: # ゼロ除算防止
            Output_full = raw_output / max_val
        else:
            Output_full = raw_output
                
        self.J = self.cfg.model.hyper_params.J
        
        # トリミング処理
        try:
            dt = self.cfg.data.dt  # サンプリング間隔 (秒)
            trim_i_seconds = self.cfg.model.hyper_params.trim_I_seconds # 入力データのトリミング秒数
            trim_o_seconds = self.cfg.model.hyper_params.trim_O_seconds # 出力データのトリミング秒数
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
                
                
        except Exception as e:
            print(f"トリミング失敗: {e}")
            self.Input = Input_full
            self.Output = Output_full
        
        base_dir = get_original_cwd()
        # self.results_dir のパスを修正し、重複した行を削除
        self.results_dir = os.path.join(base_dir, 'scripts', 'results', f'Baccus_{self.cfg.data.name}', self.date_str)
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"\n結果ファイルは {self.results_dir} に保存されます。")

    def _calculate_steady_state(self, a, kappa, b1, b2, ka, kfi, kfr, ksi, ksr):
        """
        動的モデルの初期値を定常状態から計算します。
        """
        # 定常入力 u_steady の計算 (入力=0 のときの非線形応答)
        # N_LNK.main は配列を返すので [0] を取得
        u_steady = N_LNK.main(np.array([0.0]), a, kappa, b1, b2, ka)[0]
        # 負の値にならないよう安全策（通常 erfc は正ですがパラメータ次第で念のため）
        u_steady = max(0.0, u_steady)

        # 定常状態の代数的計算
        # R を基準(1.0)として、他の状態の相対比を計算
        # 平衡式:
        #  ka * R * u = kfi * A  => A = (ka * u / kfi) * R
        #  ka * R * u = kfr * I1 => I1 = (ka * u / kfr) * R
        #  dI2 = ksi*I1 - ksr*I2*u = 0 => I2 = (ksi * I1) / (ksr * u)

        if kfi > 1e-9:
            A_ratio = (ka * u_steady) / kfi
        else:
            A_ratio = 0.0
            
        if kfr > 1e-9:
            I1_ratio = (ka * u_steady) / kfr
        else:
            I1_ratio = 0.0

        if self.use_I2 and kfr > 1e-9 and ksr > 1e-9:
             # I2 = (ksi * I1) / (ksr * u) 
             # I1 を代入すると u が約分され: I2 = (ksi * ka) / (kfr * ksr) * R
             # これにより u_steady が非常に小さい場合でも計算可能
            I2_ratio = (ksi * ka) / (kfr * ksr)
        else:
            I2_ratio = 0.0

        # 合計が1になるように正規化
        total = 1.0 + A_ratio + I1_ratio + I2_ratio
        
        # それぞれの確率（占有率）を返す
        return (1.0/total), (A_ratio/total), (I1_ratio/total), (I2_ratio/total)

    def lnk_model(self, x, save_states=False):
        """
        目的関数。与えられたパラメータxでモデルを評価します。
        初期値は定常状態計算により自動設定されます。
        """
        self.total_lnk_model_runs += 1
        try:
            hp = self.cfg.model.hyper_params
            dt = self.cfg.data.dt
            tau = hp.tau
            J = self.J
            
            # --- パラメータのアンパッキング (順番に注意) ---
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
            # シナプス後電流のパラメータ
            w_gain = x[J+10]
            w_decay = x[J+11]
            
            # ※ ここで p_R, p_A 等の初期値パラメータのアンパッキングは削除されました
            
            if not self.use_I2:
                ksi_kinetic = 0.0
                ksr_kinetic = 0.0

            # --- 定常状態を計算して初期値とする ---
            # これにより「平均的な明るさ」に順応した状態からスタートできる
            R_start, A_start, I1_start, I2_start = self._calculate_steady_state(
                a_nonlinear, kappa_nonlinear, b1_nonlinear, b2_nonlinear,
                ka_kinetic, kfi_kinetic, kfr_kinetic, ksi_kinetic, ksr_kinetic
            )
            
            # フィルタの有効長 (tau) に必要なポイント数だけを計算します。
            filter_points = int(tau / dt) + 1  # +1 は余裕を持たせるため
            
            # カーネル生成 (長さは filter_points のみ)
            linear_filter_kernel, _ = L_LNK.main(alphas, delta, filter_points, dt, tau)
            
            #畳み込みとサイズ調整
            g_full = fftconvolve(self.Input, linear_filter_kernel, mode='full')
            
            # フィルタによる位相遅れを補正するためのシフト量
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
            if np.std(u_t) < 1e-6:  # 閾値は以前より緩和
                # print("\033[31mPenalty: Saturation detected\033[0m", end='\r', flush=True)
                return 1.0 # 悪いスコア（相関1.0相当のペナルティ）として返す
            
            # Kinetic Model 
            R_state, A_state, I1_state, I2_state, W_state, check = K_LNK.main(
                len(u_t), u_t, dt, R_start, A_start, I1_start, I2_start,
                ka_kinetic, kfi_kinetic, kfr_kinetic, ksi_kinetic, ksr_kinetic, 
                w_gain, w_decay,
                label=f"LNK_run {self.total_lnk_model_runs}"
            )
            
            with open(self.debug_log_path, "a") as f:
                f.write(f"Run: {self.total_lnk_model_runs}, Check: {check}\n")
            
            # print(f"Check status for LNK model run {self.total_lnk_model_runs}: {check}", end='\r', flush=True)

            # Evaluation
            correlation = 10.0  # デフォルトのペナルティ値
            
            if check == 1:
                current_len = len(W_state)
                output_aligned = self.Output[:current_len]
    
                # DE_Simulationでは (-1 * w) と Output の相関を見ているため、ここでも -W を使う
                model_aligned = -1.0 * W_state 

                if len(output_aligned) != len(model_aligned):
                     min_l = min(len(output_aligned), len(model_aligned))
                     output_aligned = output_aligned[:min_l]
                     model_aligned = model_aligned[:min_l]

                # マスク処理: 
                # 定常状態からのスタートにより過渡応答は減るが、フィルタの遅延分などを考慮してマスクは残す
                mask_seconds = 1.0 
                mask_idx = int(mask_seconds / dt)

                if mask_idx < len(output_aligned):
                    # スライスしたデータ同士で相関を計算
                    output_eval = output_aligned[mask_idx:]
                    model_eval = model_aligned[mask_idx:]
                    
                    # --- 変更: 設定に応じて計算モジュールを切り替え ---
                    if self.objective_type == 'hybrid':
                        # 変数名を correlation に変更し、Hybridスコアを代入
                        correlation = obj_hybrid.calculate(output_eval, model_eval)
                    else:
                        # 従来のSpearman順位相関
                        correlation = obj_spearman.calculate(output_eval, model_eval)
                    # ---------------------------------------------
                        
                else:
                    correlation = 5.0 # 長さ不足時のペナルティ
            else:
                self.failed_lnk_model_runs += 1
                correlation = 5.0 # 計算失敗時のペナルティ
            
            self.current_epoch_best_fun_value = correlation

            if save_states:
                return correlation, R_state, A_state, I1_state, I2_state, W_state
            else:
                return correlation

        except Exception as e:
            print(f"エラー内容: {e}")
            import traceback
            traceback.print_exc()
            self.failed_lnk_model_runs += 1
            return 10.0  # ペナルティ値
        
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
            'ka': xk[self.J+5], 'kfi': xk[self.J+6], 'kfr': xk[self.J+7], 'ksi': xk[self.J+8], 'ksr': xk[self.J+9],
            'w_gain': xk[self.J+10], 'w_decay': xk[self.J+11]
            # p_R, p_A 等はここには含まれない
        }

        # 定常状態として計算された初期値をログ用に計算
        r_calc, a_calc, i1_calc, i2_calc = self._calculate_steady_state(
            intermediate_params['a'], intermediate_params['kappa'], intermediate_params['b1'], intermediate_params['b2'],
            intermediate_params['ka'], intermediate_params['kfi'], intermediate_params['kfr'], 
            intermediate_params['ksi'], intermediate_params['ksr']
        )
        intermediate_params['p_R_calc'] = r_calc
        intermediate_params['p_A_calc'] = a_calc
        intermediate_params['p_I1_calc'] = i1_calc
        intermediate_params['p_I2_calc'] = i2_calc
        
        # mlflow.log_metrics を使って辞書の中身を一度に記録
        # keyの先頭に "epoch_" をつけて、最終結果(optimal_)と区別する
        metrics_to_log = {f"epoch_{k}": v for k, v in intermediate_params.items()}
        metrics_to_log["epoch_correlation"] = current_best_correlation_value
        mlflow.log_metrics(metrics_to_log, step=self.epoch_counter)

        timestamp = time.strftime("%d_%H:%M:%S")
        # 表示を初期化 (行頭に戻り、行末までクリア)
        print(f"\r\033[K", end='')
        tqdm.write(f"---{timestamp} | Epoch {self.epoch_counter:03d} | Cost: {self.current_epoch_best_fun_value:.4f} ---")

    def save_optimal_results(self, optimal_params, optimal_correlation, R_state, A_state, I1_state, I2_state, W_state):
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
            # 初期値パラメータはリストから除外
            param_keys = [
                'delta', 'a', 'kappa', 'b1', 'b2', 
                'ka', 'kfi', 'kfr', 'ksi', 'ksr',
                'w_gain', 'w_decay'
            ]
            
            for key in param_keys:
                param_map[key] = optimal_params[idx]
                idx += 1
                
            # 相関係数もマップに追加
            param_map['correlation'] = optimal_correlation

            # --- 計算された定常状態（初期状態）の保存 ---
            R_calc, A_calc, I1_calc, I2_calc = self._calculate_steady_state(
                param_map['a'], param_map['kappa'], param_map['b1'], param_map['b2'],
                param_map['ka'], param_map['kfi'], param_map['kfr'], 
                param_map['ksi'], param_map['ksr']
            )

            param_map['R_start_calculated'] = R_calc
            param_map['A_start_calculated'] = A_calc
            param_map['I1_start_calculated'] = I1_calc
            param_map['I2_start_calculated'] = I2_calc

            # ---ファイル保存ループ (エラーハンドリング付き) ---
            saved_count = 0
            for name, val in param_map.items():
                try:
                    # 個別のファイル保存に失敗しても他は保存し続ける
                    save_results(val, os.path.join(self.results_dir, f'{name}.txt'))
                    saved_count += 1
                except Exception:
                    pass

            # --- MLflowへの記録 ---
            try:
                final_metrics = {f"optimal_{k}": v for k, v in param_map.items()}
                mlflow.log_metrics(final_metrics)
            except Exception:
                pass

            # --- 時系列状態変数の保存 ---
            state_dir = os.path.join(self.results_dir, 'state')
            os.makedirs(state_dir, exist_ok=True)
            
            save_results(R_state, os.path.join(state_dir, 'R_state.txt'))
            save_results(A_state, os.path.join(state_dir, 'A_state.txt'))
            save_results(I1_state, os.path.join(state_dir, 'I1_state.txt'))
            save_results(I2_state, os.path.join(state_dir, 'I2_state.txt'))
            save_results(W_state, os.path.join(state_dir, 'W_state.txt'))
            
            print("すべての保存処理が完了しました。")
            
        except Exception as e:
            print(f"Error saving results: {e}")
            
    def run(self):
        """
        最適化プロセスを実行します。
        """
        # ワークステーションでの並列処理の際にNumbaのJITが渋滞する問題を回避するためのウォームアップ
        print("Numba JITコンパイラのウォームアップ中...")
        try:
            # ダミーのパラメータ配列 (長さ: J + 12) を作成
            x_dummy = np.ones(self.J + 12) 
            # 目的関数を一度だけ実行して、コンパイルを強制する
            self.lnk_model(x_dummy, save_states=False)
            print("ウォームアップ完了。最適化を開始します。")
        except Exception as e:
            print(f"警告: ウォームアップ中にエラーが発生しました: {e}")
            # エラーが起きても、本番の最適化は続行してみる
            
        # Configからパラメータ境界(param_bounds)を取得
        pb = self.cfg.model.hyper_params.param_bounds
        J = self.J
        
        # alphas (L1-LJ) の境界を生成
        alpha_bounds_tuple = tuple(pb.LinearFilter.alphas)
        try_bounds = [alpha_bounds_tuple] * J
        
        #Configのリスト [min, max] を tuple(min, max) に変換
        try:
            try_bounds.extend([
                tuple(pb.LinearFilter.delta),
                tuple(pb.Nonlinear.a),
                tuple(pb.Nonlinear.kappa),
                tuple(pb.Nonlinear.b1),
                tuple(pb.Nonlinear.b2),
                tuple(pb.Kinetics.ka),
                tuple(pb.Kinetics.kfi),
                tuple(pb.Kinetics.kfr),
                tuple(pb.Kinetics.ksi),
                tuple(pb.Kinetics.ksr),
                # ★ 追加されたパラメータの範囲
                tuple(pb.Kinetics.w_gain),
                tuple(pb.Kinetics.w_decay),
                # 初期状態 p_R, p_A, p_I1, p_I2 は削除
            ])
        except Exception as e:
            print(f"Config Error: {e}")
            raise

        
        #use_I2=False の場合、探索範囲を [0, 0] に固定
        if not self.use_I2:
            print("I2が無効なため、ksi, ksr の探索範囲を [0.0, 0.0] に固定します。")
            # インデックスもずれるので注意: ksrは J+9
            try_bounds[J + 8] = (0.0, 0.0) # ksi
            try_bounds[J + 9] = (0.0, 0.0) # ksr
            # p_I2 の固定処理は不要になりました

        param_names = [f'L{i+1}' for i in range(self.J)] + [
            'delta', 'a', 'kappa', 'b1', 'b2', 'ka', 'kfi', 'kfr', 'ksi', 'ksr',
            'w_gain', 'w_decay' # p_R, p_A... 削除
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
            print(f"警告: 'optimization.strategy' の設定が不正です。デフォルトの 'best1bin' を使用します。")
            print(f"詳細: {e}")
            de_strategy_str = 'best1bin'

        # --- 最適化の実行 ---
        try:
            # 実際の最適化処理
            result = differential_evolution(
                self.lnk_model, 
                bounds=try_bounds, 
                strategy=de_strategy_str,
                maxiter=opt_cfg.maxiter, 
                popsize=opt_cfg.popsize, 
                tol=1e-6, 
                mutation=(0.5, 1.0), 
                recombination=0.7, 
                disp=True,
                callback=self.save_intermediate_results,
                workers=opt_cfg.workers,
                updating=opt_cfg.updating
            )
            
            print(f"\n最適化完了。終了コード: {result.message}")
            
            # 最適化されたパラメータ
            optimal_params = result.x
            
            # 最終的なモデルの状態を取得して保存
            # ここでは最終スコア(correlation)を返り値から取得
            optimal_correlation, R_final, A_final, I1_final, I2_final, W_final = self.lnk_model(optimal_params, save_states=True)
            
            # 結果の保存
            self.save_optimal_results(optimal_params, optimal_correlation, R_final, A_final, I1_final, I2_final, W_final)

            # 実行終了時にMLflowの実行を明示的に終了（通常はwith文で管理するが念のため）
            mlflow.end_run()

        except KeyboardInterrupt:
            print("\nユーザーによる中断。現在の最良の結果を保存します...")
            # 中断時でもそれまでの最良の結果があれば保存したい場合の処理をここに記述可能
            # 現状はそのまま終了
            pass
        except Exception as e:
            print(f"最適化実行中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
                
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
    
    # I2の使用有無をラベル化
    use_I2_str = "I2_ON" if cfg.hyper_params.get('use_I2', True) else "I2_OFF"
    
    # 以前の行で作成した strategy_str_for_name を利用する
    # 例: rand/1/exp_I2_ON_20231025_1200
    run_name = f"{strategy_str_for_name}_{use_I2_str}_{time.strftime('%Y%m%d_%H%M')}"

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