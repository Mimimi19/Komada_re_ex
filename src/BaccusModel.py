# coding: utf-8
from tqdm import tqdm
import pprint
import math
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import differential_evolution
import yaml
import os
import time

import components.F_LNX as F_LNX
import components.N_LNK as N_LNK
import components.K_baccus as K_LNK

# グローバルカウンタ
total_lnk_model_runs = 0
failed_lnk_model_runs = 0
date_str = time.strftime("%Y%m%d_%H")

# Docker環境では /app/src となる
script_dir = os.path.dirname(os.path.abspath(__file__))
# プロジェクトのルートディレクトリを取得 (Docker環境では /app となる)
project_root_dir = os.path.dirname(script_dir)

# エポックごとの保存のためのグローバル変数
# current_epoch_best_fun_value は LNK_model が計算した最新の目的関数値を保持
current_epoch_best_fun_value = 1000.0 # 最小化問題なので初期値は大きな値
epoch_counter = 0 # エポックカウンター

# 提供データのインプット
data_options =  "cb1"
# data_options =  "cb2"
if data_options == "cb1":
    print("cb1のデータを使用します")
    #cb1データ
    Input = np.genfromtxt(os.path.join(script_dir, "components", "Provided_Data", "cb1", "wn_0.0002s.txt"))
    Output = np.genfromtxt(os.path.join(script_dir, "components", "Provided_Data", "cb1", "cb1_Fourier_Result.txt"))
elif data_options == "cb2":
    print("cb2のデータを使用します")
    #cb2データ

    Input = np.genfromtxt(os.path.join(script_dir, "components", "Provided_Data", "cb2", "wn.txt")) # wn.txtが正しいか確認
    Output = np.genfromtxt(os.path.join(script_dir, "components", "Provided_Data", "cb2", "cb2_Fourier_Result.txt"))
# 設定ファイルの読み込み
def load_config(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# 実験結果の保存 (NumPy配列の保存に対応するため修正)
def save_results(data, filepath):
    # print(f"結果を保存: {filepath}") 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if isinstance(data, np.ndarray):
        # NumPy配列はnp.savetxtで保存
        np.savetxt(filepath, data, fmt='%.6f')
    else:
        # その他のデータは通常のファイル書き込み
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(str(data) + '\n')
# 目的関数定義
# save_states引数を追加
def LNK_model(x, save_states=False):
    global total_lnk_model_runs
    global failed_lnk_model_runs
    global current_epoch_best_fun_value # グローバル変数を参照

    total_lnk_model_runs += 1 # 関数の開始時に合計実行回数をインクリメント
    #ハイパーパラメータの設定
    config_file_path = os.path.join(script_dir, "components", "config", "Baccus.yaml")
    # 設定ファイルからパラメータを取得
    try:
        # 設定を読み込む
        config = load_config(config_file_path)
        # 設定ファイルからパラメータを取得
        try:
            dt = config['dt'] #刻み幅
            #初期値設定のための初期値
            R_start = config['R_start']
            A_start = config['A_start']
            I1_start = config['I_start'] # Assuming I_start corresponds to I1_start
            I2_start = 0.0 # Assuming I2_start is 0.0 as it's not in config
            
            tau = config['tau']
            #基底関数の数
            J = config['J'] # 基底関数の数 (J)
        except KeyError as e:
            print(f"設定ファイルに必要なキーがありません: {e}")
            raise
    except FileNotFoundError:
        print(f"エラー: '{config_file_path}' が見つかりません。")
        raise
    except yaml.YAMLError as exc:
        print(f"YAMLファイルのパースエラー: {exc}")
        raise
    except KeyError as e:
        print(f"設定ファイルに予期せぬキーがありません: {e}")
        raise
   
    # 線形フィルタパラメータ (x[0]からx[J-1])
    # x[J] は delta
    # x[J+1]からx[J+3-1] は 非線形パラメータ (a, b1, b2)
    # x[J+3]からx[J+3+3-1] は 動的パラメータ (ka, kfi, kfr)
    
    alphas = x[0:J] # x[0]からx[J-1]までのJ個のパラメータ
    delta = x[J] # x[J]
    a_nonlinear = x[J+1] # x[J+1]
    b1_nonlinear = x[J+2] # x[J+2]
    b2_nonlinear = x[J+3] # x[J+3]
    ka_kinetic = x[J+4] # x[J+4]
    kfi_kinetic = x[J+5] # x[J+5]
    kfr_kinetic = x[J+6] # x[J+6]

    # ksiとksrは現在定数として扱われている
    ksi_kinetic = 0.0
    ksr_kinetic = 0.0

    #Linear Filterについて
    # F_LNX.mainの戻り値は、線形フィルターカーネル全体と時間軸
    # print("線形フィルターの計算を開始します...") # 最適化中は頻繁な出力は抑制
    Linear_Filter_kernel, _ = F_LNX.main(alphas, delta, dt, tau, J)

    #畳み込みでチルダgの作成
    # mode='full'の場合、出力は len(Input) + len(Linear_Filter_kernel) - 1 の長さになる
    # モデルの出力長を合わせるために、適切なスライスが必要になる場合がある
    tild_g_full = np.convolve(Input, Linear_Filter_kernel, mode='full')

    # モデルの時間ステップ数に合わせるため、入力信号の長さに合わせる
    # ここでは、Inputの長さ (80000) に合わせる
    g_len = len(Input)
    tild_g = tild_g_full[:g_len] # 適切な長さにスライス

    #スケーリング定数を求めるためのパラメータ
    Record1 = 0
    Record2 = 0
    
    #チルダgの計算（スケーリング前）
    # Inputとtild_gは同じ長さである必要がある
    if len(tild_g) != len(Input):
        print(f"Error: Length of tild_g ({len(tild_g)}) does not match length of Input ({len(Input)}) for scaling.")
        raise
    # 入力刺激とチルダgの分散を求める
    # dtは合計の計算で考慮される
    Record1 = np.sum(tild_g * tild_g) * dt
    Record2 = np.sum(Input * Input) * dt
    
    #スケーリング係数を求める
    if Record2 == 0:
        print("Error: Record2 is zero, cannot calculate scale_Linear. Input might be all zeros.")
        raise
    scale_Linear = math.sqrt(Record1 / Record2)
    
    #スケーリング
    g = tild_g / scale_Linear
    
    #Nonlinearモデル
    # print("非線形モデルの計算を開始します...")
    U_Nonlinear = np.array([N_LNK.main(val, a_nonlinear, b1_nonlinear, b2_nonlinear) for val in tqdm(g, leave=False, desc="N_Model")])
        
    #Kineticモデル
    # K_LNK.mainの引数を修正: time_steps, u_input, dt, R_start, A_start, I1_start, I2_start, ka, kfi, kfr, ksi, ksr
    # print("Kineticモデルの計算を開始します...")
    R_state, A_state, I1_state, I2_state ,check= K_LNK.main(
        len(U_Nonlinear), U_Nonlinear, dt, R_start, A_start, I1_start, I2_start,
        ka_kinetic, kfi_kinetic, kfr_kinetic, ksi_kinetic, ksr_kinetic,
        label=f"LNK_run {total_lnk_model_runs}" # K_baccusのtqdmのdescに表示されるラベル
    )

    # print("スピアマンの相関係数を計算します...")
    #スピアマンによる評価
    correlation = 1000.0 # デフォルトで大きな値を設定
    if check == 1:
        # OutputとResultの長さを合わせる必要がある
        # A_stateはU_Nonlinearと同じ長さになるはず
        keep_Post = (-1) * A_state[:len(Output)] # Outputの長さに合わせる
        
        # Outputとkeep_Postの長さを確認
        if len(Output) != len(keep_Post):
            print(f"Warning: Length of Output ({len(Output)}) and calculated Post-synaptic potential ({len(keep_Post)}) do not match.")
            # For correlation calculation, trim to the shorter length
            min_len = min(len(Output), len(keep_Post))
            Output_trimmed = Output[:min_len]
            keep_Post_trimmed = keep_Post[:min_len]
            correlation, pvalue = spearmanr(Output_trimmed, keep_Post_trimmed)
        else:
            correlation, pvalue = spearmanr(Output, keep_Post)
        


        # 最小化問題のため相関の負の値を返す
        correlation = (-1) * correlation 
    else:
        print("Kinetic model が失敗しました.")
        failed_lnk_model_runs += 1 # Kineticモデルが失敗した場合は失敗としてマーク
        correlation = 1000.0 # 状態が不正な場合は大きなペナルティ
    # 結果の表示
    # print(f"相関係数: {correlation:.4f}")
    # LNK_modelの進捗表示 (total_lnk_model_runs が 100 の倍数または初回のみ)
    if total_lnk_model_runs % 100 == 0 or total_lnk_model_runs == 1:
        if total_lnk_model_runs > 0:
            current_failure_rate = (failed_lnk_model_runs / total_lnk_model_runs) * 100
            tqdm.write(f"LNK_model の失敗回数/合計実行回数(失敗率): {failed_lnk_model_runs}/{total_lnk_model_runs} ({current_failure_rate:.2f}%)")
        else:
            tqdm.write("実行記録がありません。")
            
    # コールバック関数で参照できるように、現在の目的関数値をグローバル変数に格納
    global current_epoch_best_fun_value
    current_epoch_best_fun_value = correlation

    # save_statesがTrueの場合のみ、相関係数と状態を返す
    if save_states:
        return correlation, R_state, A_state, I1_state, I2_state
    else:
        return correlation

# 最適化結果を保存する関数 (最終結果用)
def save_optimal_results(optimal_params, optimal_correlation, R_state, A_state, I1_state, I2_state, J):
    """
    最適化されたパラメータと対応する状態をファイルに保存します。
    この関数は最適化完了後に一度だけ呼び出されます。
    """
    results_base_dir = os.path.join(project_root_dir, 'results', 'Baccus_'+ data_options)

    # パラメータ用のディレクトリ (日付と時刻でユニークに)
    param_results_dir = os.path.join(results_base_dir, date_str)
    os.makedirs(param_results_dir, exist_ok=True)

    print(f"\n最適化結果を {param_results_dir} に保存中...")

    for i in range(J): # 線形フィルタのパラメータを保存
        save_results(optimal_params[i], os.path.join(param_results_dir, f'L{i+1}.txt'))
    save_results(optimal_params[J], os.path.join(param_results_dir, 'delta.txt'))
    save_results(optimal_params[J+1], os.path.join(param_results_dir, 'a.txt'))
    save_results(optimal_params[J+2], os.path.join(param_results_dir, 'b1.txt'))
    save_results(optimal_params[J+3], os.path.join(param_results_dir, 'b2.txt'))
    save_results(optimal_params[J+4], os.path.join(param_results_dir, 'ka.txt'))
    save_results(optimal_params[J+5], os.path.join(param_results_dir, 'kfi.txt'))
    save_results(optimal_params[J+6], os.path.join(param_results_dir, 'kfr.txt'))
    save_results(optimal_correlation, os.path.join(param_results_dir, 'correlation.txt'))

    # 状態の保存 (最終結果としてのみ)
    state_results_dir = os.path.join(param_results_dir, 'state') # パラメータディレクトリの下にstateディレクトリを作成
    os.makedirs(state_results_dir, exist_ok=True)
    print(f"状態を {state_results_dir} に保存中...")
    save_results(R_state, os.path.join(state_results_dir, 'R_state.txt'))
    save_results(A_state, os.path.join(state_results_dir, 'A_state.txt'))
    save_results(I1_state, os.path.join(state_results_dir, 'I1_state.txt'))
    save_results(I2_state, os.path.join(state_results_dir, 'I2_state.txt'))
    print("保存が完了しました。")

# エポックごとに結果を保存するコールバック関数
def save_intermediate_results(xk, convergence):
    """
    differential_evolutionの各イテレーション（エポック）の終わりに呼び出され、
    現在の最良パラメータと相関係数を保存します。
    状態データは保存しません。
    """
    global epoch_counter
    global current_epoch_best_fun_value # LNK_modelで更新された最新の目的関数値

    epoch_counter += 1
    
    # 中間結果を保存するディレクトリ
    intermediate_results_dir = os.path.join(project_root_dir, 'results', 'Baccus_'+ data_options, date_str, 'epochs')
    os.makedirs(intermediate_results_dir, exist_ok=True)

    # 現在の最良パラメータを保存
    params_filepath = os.path.join(intermediate_results_dir, f'epoch_{epoch_counter:03d}_params.txt')
    save_results(xk, params_filepath)

    # 現在の最良相関係数を保存 (LNK_modelで更新されたグローバル変数を使用)
    correlation_filepath = os.path.join(intermediate_results_dir, f'epoch_{epoch_counter:03d}_correlation.txt')
    save_results(-current_epoch_best_fun_value, correlation_filepath) # 負の値を正に戻して保存

    tqdm.write(f"--- Epoch {epoch_counter:03d} Results Saved (Correlation: {-current_epoch_best_fun_value:.4f}) ---")


def main(Try_bounds):
    #差分進化法
    # disp=True で進捗を表示
    # updating='deferred' で更新を遅延させる
    # maxiter=100 で最大反復回数を設定
    # popsize=200 で個体群のサイズを設定
    # strategy='rand1bin' で戦略を設定
    # workers=-1 で全てのCPUコアを使用
    # 差分進化アルゴリズムの目的関数呼び出し: 約 20,000 回 (maxiter * popsize)
    # callback引数に中間結果保存関数を指定
    print("差分進化法による最適化を開始します...")
    result = differential_evolution(LNK_model, Try_bounds, disp=True, updating = 'deferred', maxiter = 100, popsize = 200, strategy = 'rand1bin', workers=-1, callback=save_intermediate_results)

    #表示
    print("\n最適化が完了しました。")
    pprint.pprint(result)

    # 最適なパラメータを取得
    optimal_params = result.x
    optimal_correlation_value = -result.fun # 最小化された負の相関係数を正に戻す

    # 最適なパラメータでLNK_modelを再度実行し、状態を取得
    print("\n最終的な状態を取得するため、最適なパラメータでモデルを再実行します...")
    # LNK_modelの戻り値がタプルであることを考慮してアンパック
    final_correlation_check, R_state_final, A_state_final, I1_state_final, I2_state_final = LNK_model(optimal_params, save_states=True)

    # Jの値は設定ファイルから取得する必要がある
    config_file_path = os.path.join(script_dir, "components", "config", "Baccus.yaml")
    try:
        config = load_config(config_file_path)
        J = config['J']
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        print(f"設定ファイルの読み込みまたはキーの取得エラー: {e}")
        print("Jの値が取得できないため、最終結果の保存をスキップします。")
        return

    # 状態が正常に取得できた場合のみ保存
    if A_state_final is not None:
        save_optimal_results(optimal_params, optimal_correlation_value,
                             R_state_final, A_state_final, I1_state_final, I2_state_final, J)
    else:
        print("Kineticモデルが最終実行で失敗したため、状態は保存されません。")
        print(f"最終的な相関係数: {optimal_correlation_value:.4f}")


if __name__ == "__main__":
    #探索するパラメータの範囲
    #左からlinearのalphas 15個, delta, 非線形パラメータ3個 (a, b1, b2), 動的パラメータ3個 (ka, kfi, kfr)
    # total parameters: 15 (alphas) + 1 (delta) + 3 (nonlinear) + 3 (kinetic) = 22 parameters
    
    # J = 15 alphas
    # x[0] to x[14] for alphas
    # x[15] for delta
    # x[16] for a
    # x[17] for b1
    # x[18] for b2
    # x[19] for ka
    # x[20] for kfi
    # x[21] for kfr

    Try_bounds = [
        (0.01, 1.0) for _ in range(15) # alphas (L1-L15)
    ] + [
        (0.05, 0.2), # delta
        (0.1, 10.0), # a (nonlinear)
        (-5.0, 5.0), # b1 (nonlinear)
        (-10.0, 10.0),# b2 (nonlinear)
        (0.01, 1.0), # ka (kinetic)
        (0.01, 1.0), # kfi (kinetic)
        (0.01, 1.0)  # kfr (kinetic)
    ]
    print(f"Number of parameters in Try_bounds: {len(Try_bounds)}")
    main(Try_bounds)
