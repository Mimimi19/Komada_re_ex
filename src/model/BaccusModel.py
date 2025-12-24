# src/model/BaccusModel.py
# -*- coding: utf-8 -*-
import sys
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
from hydra.core.hydra_config import HydraConfig
import mlflow
import requests
from dotenv import load_dotenv

# import sys と import os の直後に配置
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/model
src_dir = os.path.dirname(current_dir) # src

# srcディレクトリをパスに追加して components を読み込めるようにする
if src_dir not in sys.path:
    sys.path.append(src_dir)

import components.L_LNK as L_LNK
import components.N_LNK as N_LNK
import components.K_baccus as K_LNK

# 目的関数モジュールのインポート 
import components.objectives.spearman as obj_spearman
import components.objectives.hybrid as obj_hybrid

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
        # config.yaml に objective_type がない場合は 'hybrid' を使用
        self.objective_type = self.cfg.hyper_params.get('objective_type', 'hybrid')
        
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
            print("\nI2 state (ksi, ksr) は最適化対象に含まれます。\n") # p_I2削除に伴いメッセージ修正

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
                
        self.J = self.cfg.hyper_params.J
        
        # トリミング処理
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
                
                
        except Exception as e:
            print(f"トリミング失敗: {e}")
            self.Input = Input_full
            self.Output = Output_full
        
        try:
            # hydra.run.dir で指定されたパスを確実に取得する
            self.results_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            # 万が一Hydra経由でない場合のフォールバック
            self.results_dir = os.getcwd()
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
            hp = self.cfg.hyper_params
            dt = self.cfg.data.dt
            tau = hp.tau  # 大局（低周波）用
            tau_short = hp.get('tau_short', None)  # 局所（高周波）用（無ければNone）
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
            
            if not self.use_I2:
                ksi_kinetic = 0.0
                ksr_kinetic = 0.0

            # --- 定常状態を計算して初期値とする ---
            # これにより「平均的な明るさ」に順応した状態からスタートできる
            R_start, A_start, I1_start, I2_start = self._calculate_steady_state(
                a_nonlinear, kappa_nonlinear, b1_nonlinear, b2_nonlinear,
                ka_kinetic, kfi_kinetic, kfr_kinetic, ksi_kinetic, ksr_kinetic
            )
            # Linear Model
            # --- 大局（低周波）用の長窓 ---
            filter_points = int(tau / dt) + 1  # +1 は余裕を持たせるため

            # カーネル生成 (長さは filter_points のみ)
            linear_filter_kernel, _ = L_LNK.main(alphas, delta, filter_points, dt, tau)

            # 畳み込みとサイズ調整
            g_full = fftconvolve(self.Input, linear_filter_kernel, mode='full')

            # フィルタによる位相遅れを補正するためのシフト量
            shift_idx = int(tau / dt)

            # データ長に合わせて切り出し
            if len(g_full) > shift_idx + len(self.Input):
                g_long = g_full[shift_idx : shift_idx + len(self.Input)]
            else:
                # 万が一長さが足りない場合のフォールバック（後ろをゼロ埋めなど）
                g_long = np.zeros(len(self.Input))
                take = max(0, min(len(g_full) - shift_idx, len(self.Input)))
                if take > 0:
                    g_long[:take] = g_full[shift_idx:shift_idx+take]

            # 正規化（飽和回避）。長窓は大局の形を担う
            g_long_std = np.std(g_long)
            if g_long_std > 1e-9:
                g_long = g_long / g_long_std

            # --- 局所（高周波）用の短窓（任意） ---
            # tau_short が設定されている場合のみ、短窓フィルタも同時に使う
            if tau_short is not None and tau_short > 0:
                
                filter_points_s = int(tau_short / dt) + 1
                #短窓は点数が少ないので、使える基底数を制限する
                J_eff_s = min(len(alphas), filter_points_s)
                alphas_s = alphas[:J_eff_s]

                linear_filter_kernel_s, _ = L_LNK.main(alphas_s, delta, filter_points_s, dt, tau_short)


                g_full_s = fftconvolve(self.Input, linear_filter_kernel_s, mode='full')
                shift_idx_s = int(tau_short / dt)

                if len(g_full_s) > shift_idx_s + len(self.Input):
                    g_short = g_full_s[shift_idx_s : shift_idx_s + len(self.Input)]
                else:
                    g_short = np.zeros(len(self.Input))
                    take_s = max(0, min(len(g_full_s) - shift_idx_s, len(self.Input)))
                    if take_s > 0:
                        g_short[:take_s] = g_full_s[shift_idx_s:shift_idx_s+take_s]

                # 正規化（飽和回避）。短窓は局所の細部を担う
                g_short_std = np.std(g_short)
                if g_short_std > 1e-9:
                    g_short = g_short / g_short_std

                # 合成（最小変更：単純和）
                # g_t = g_long + beta_short*g_short # beta_short を導入しても良いが、まずは単純和で試す
                g_t = g_long + g_short
            else:
                g_t = g_long

            # これがないと g_t が ±100 になり、非線形関数が飽和する
            g_std = np.std(g_t)
            if g_std > 1e-9:
                g_t = g_t / g_std
                
            # Nonlinear Model
            u_t = N_LNK.main(g_t, a_nonlinear, kappa_nonlinear, b1_nonlinear, b2_nonlinear, ka_kinetic)
            # 飽和ペナルティ 
            # u_t の標準偏差が極端に小さい（平坦）、または値が張り付いている場合にペナルティ
            if np.std(u_t) < 1e-6:  # 閾値は以前より緩和
                print("\033[31mPenalty: Saturation detected\033[0m", end='\r', flush=True)
                penalty_val = 10.0 # ペナルティを強化 (1.0 -> 10.0)
                if save_states:
                    return penalty_val, None, None, None, None
                return penalty_val
            
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
            score = 5.0  # デフォルトペナルティ（大きいほど悪い）

            if check == 1:
                # 配列長の調整
                current_len = len(A_state)
                output_aligned = self.Output[:current_len]

                # 先行研究寄り: A_state を出力として扱う
                model_aligned = A_state

                # スケール差でスコアが支配されないように z-score 正規化
                o_std = np.std(output_aligned)
                m_std = np.std(model_aligned)
                if o_std > 1e-9:
                    output_aligned = (output_aligned - np.mean(output_aligned)) / o_std
                if m_std > 1e-9:
                    model_aligned = (model_aligned - np.mean(model_aligned)) / m_std

                if len(output_aligned) != len(model_aligned):
                    min_l = min(len(output_aligned), len(model_aligned))
                    output_aligned = output_aligned[:min_l]
                    model_aligned = model_aligned[:min_l]

                # マスク処理（先頭のトランジェントを除外）
                mask_seconds = 1.0
                mask_idx = int(mask_seconds / dt)

                if mask_idx < len(output_aligned):
                    output_eval = output_aligned[mask_idx:]
                    model_eval = model_aligned[mask_idx:]

                    # --- 目的関数の選択 ---
                    if self.objective_type in ('hybrid', 'band_low_only', 'band_main_only', 'band_full'):
                        # obj_hybrid は「最小化すべきスコア」を返す実装（相関は内部でマイナス符号で最大化）
                        correlation = obj_hybrid.calculate(
                            output_eval,
                            model_eval,
                            dt=dt,
                            objective_type=self.objective_type,
                            # hybrid で band(main) を混ぜたい場合だけ w_band を >0 にする
                            # band_* の場合は w_band は無視されます
                            w_band=hp.get('w_band', 0.0),
                        )
                    else:
                        # Spearman順位相関（従来）
                        correlation = obj_spearman.calculate(output_eval, model_eval)
                else:
                    score = 5.0  # 長さ不足時のペナルティ
            else:
                self.failed_lnk_model_runs += 1
                score = 5.0  # 計算失敗時のペナルティ

            # この関数の返り値は最小化対象（score）
            correlation = score
            


            
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
            return 5.0  # ペナルティ値
        
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
        tqdm.write(f"---{timestamp} | Epoch {self.epoch_counter:03d} | Corr: {current_best_correlation_value:.4f} ---")

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
            # 初期値パラメータはリストから除外
            param_keys = [
                'delta', 'a', 'kappa', 'b1', 'b2', 
                'ka', 'kfi', 'kfr', 'ksi', 'ksr',
                
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
            # ダミーのパラメータ配列 (長さ: J + 10) を作成
            x_dummy = np.ones(self.J + 10) 
            # 目的関数を一度だけ実行して、コンパイルを強制する
            self.lnk_model(x_dummy, save_states=False)
            print("ウォームアップ完了。最適化を開始します。")
        except Exception as e:
            print(f"警告: ウォームアップ中にエラーが発生しました: {e}")
            # エラーが起きても、本番の最適化は続行してみる
            
        # Configからパラメータ境界(param_bounds)を取得
        pb = self.cfg.hyper_params.param_bounds
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
                # （シナプス後電流を廃止）
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
            'delta', 'a', 'kappa', 'b1', 'b2', 'ka', 'kfi', 'kfr', 'ksi', 'ksr'
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
        
        #局所探索 (Powell)
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

        # 戻り値受け取り変更
        res_tuple = self.lnk_model(optimal_params, save_states=True)

        if isinstance(res_tuple, tuple) and len(res_tuple) == 5:
            _, r_final, a_final, i1_final, i2_final = res_tuple
            self.save_optimal_results(optimal_params, optimal_correlation, r_final, a_final, i1_final, i2_final)
            mlflow.log_artifacts(self.results_dir, artifact_path="results")
        else:
            print("Kineticモデルが最終実行で失敗したため、状態は保存されません。")
            print(f"最終的な相関係数: {optimal_correlation:.4f}")
            mlflow.log_metric("final_correlation_on_failure", optimal_correlation)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
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
