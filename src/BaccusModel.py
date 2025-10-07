# coding: utf-8
from tqdm import tqdm
import pprint
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import differential_evolution
import yaml
import os
import time
import sys


import components.F_LNK  as F_LNK
import components.N_LNK as N_LNK
import components.K_baccus as K_LNK

def load_config(filepath):
    """
    ハイパーパラメータ設定ファイルを読み込みます。
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


# Docker環境では /app/src となる
script_dir = os.path.dirname(os.path.abspath(__file__))
# プロジェクトのルートディレクトリを取得 (Docker環境では /app となる)
project_root_dir = os.path.dirname(script_dir)

# スクリプトのディレクトリを取得 (config_file_pathの構築に必要)
config_file_path = os.path.join(script_dir, "..", "config", "Baccus.yaml")

# 設定ファイルの読み込み (グローバルスコープで一度だけ読み込む)
try:
    # 設定を読み込む
    config = load_config(config_file_path)
except FileNotFoundError:
    print(f"エラー: '{config_file_path}' が見つかりません。") # printに変更
    raise
except yaml.YAMLError as exc:
    print(f"YAMLファイルのパースエラー: {exc}") # printに変更
    raise
except KeyError as e:
    print(f"設定ファイルに予期せぬキーがありません: {e}") # printに変更
    raise

# グローバルカウンタ
total_lnk_model_runs = 0
failed_lnk_model_runs = 0
date_str = time.strftime("%Y%m%d_%H")


# エポックごとの保存のためのグローバル変数
# current_epoch_best_fun_value は LNK_model が計算した最新の目的関数値を保持
current_epoch_best_fun_value = 1000.0 # 最小化問題なので初期値は大きな値
epoch_counter = 0 # エポックカウンター

#基底関数の数
J = config['J'] # 基底関数の数 (J)
# 提供データのインプット
data_options = config['data_options']

if data_options == "cb1":
    print("cb1のデータを使用します")
    #cb1データ
    Input = np.genfromtxt(os.path.join(script_dir, "..", "data", "cb1", "wn_0.0002s.txt"))
    Output = np.genfromtxt(os.path.join(script_dir, "..", "data", "cb1", "average80-86.txt"))
elif data_options == "cb2":
    print("cb2のデータを使用します")
    #cb2データ
    Input = np.genfromtxt(os.path.join(script_dir, "..", "data", "cb2", "wn.txt"))
    Output = np.genfromtxt(os.path.join(script_dir, "..", "data", "cb2", "average82-88.txt"))


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

    try:
        dt = config['dt'] #刻み幅
        #初期値設定のための初期値
        R_start = config['R_start']
        A_start = config['A_start']
        I1_start = config['I1_start'] # Assuming I1_start corresponds to I1_start
        I2_start = config['I2_start'] # Assuming I2_start corresponds to I2_start

        tau = config['tau']
    except KeyError as e:
        print(f"設定ファイルに必要なキーがありません: {e}")
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
    ksi_kinetic = x[J+7] # x[J+7]
    ksr_kinetic = x[J+8] # x[J+8]

    t = min(len(Input), len(Output)) #データの長さ
    #Linear Filterについて
    # F_LNK.mainの戻り値は、線形フィルターカーネル全体と時間軸
    # print("線形フィルターの計算を開始します...") # 最適化中は頻繁な出力は抑制
    Linear_Filter_kernel, _ = F_LNK.main(alphas, delta, t, dt, tau) #linear filter,時間軸
    #畳み込みでg(t)の作成
    g_t = np.convolve(Input[:t], Linear_Filter_kernel, mode='same')


    #スケーリング定数を求めるためのパラメータ
    Record1 = 0
    Record2 = 0

    #g(t)の計算（スケーリング前）
    # Inputとg_tは同じ長さである必要がある
    if len(g_t) != t:
        print(f"Error: Length of g_t ({len(g_t)}) does not match length of Input ({t}) for scaling.")
        raise
 
    #Nonlinearモデル
    # print("非線形モデルの計算を開始します...")
    u_t = N_LNK.main(g_t, a_nonlinear, b1_nonlinear, b2_nonlinear)
    #Kineticモデル
    # K_LNK.mainの引数を修正: time_steps, u_input, dt, R_start, A_start, I1_start, I2_start, ka, kfi, kfr, ksi, ksr
    # print("Kineticモデルの計算を開始します...")
    R_state, A_state, I1_state, I2_state ,check= K_LNK.main(
        len(u_t), u_t, dt, R_start, A_start, I1_start, I2_start,
        ka_kinetic, kfi_kinetic, kfr_kinetic, ksi_kinetic, ksr_kinetic,
        label=f"LNK_run {total_lnk_model_runs}" # K_baccusのtqdmのdescに表示されるラベル
    )

    # print("スピアマンの相関係数を計算します...")
    #スピアマンによる評価
    correlation = 1.0 # 初期化
    if check == 1:
        # OutputとResultの長さを合わせる必要がある
        # A_stateはU_Nonlinearと同じ長さになるはず
        keep_Post = A_state[:t]
        Output_trimmed = Output[:t]
        correlation, pvalue = spearmanr(Output_trimmed, keep_Post)



        # 最小化問題のため相関の負の値を返す
        correlation = (-1) * correlation
    else:
        # print("Kinetic model が失敗しました.")
        failed_lnk_model_runs += 1 # Kineticモデルが失敗した場合は失敗としてマーク
        correlation = 1.0 # 状態が不正な場合は大きなペナルティ
    # 結果の表示
    # print(f"相関係数: {correlation:.4f}")

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
def save_intermediate_results(xk):
    """
    differential_evolutionの各イテレーション（エポック）の終わりに呼び出され、
    現在の最良パラメータと相関係数を保存します。
    状態データは保存しません。
    """
    global epoch_counter
    global current_epoch_best_fun_value # LNK_modelで更新された最新の目的関数値
    global total_lnk_model_runs

    epoch_counter += 1

    # 中間結果を保存するディレクトリ
    intermediate_results_dir = os.path.join(project_root_dir, '..','scripts','results',  data_options , date_str + '_Baccus', 'epochs')
    os.makedirs(intermediate_results_dir, exist_ok=True)

    # 現在の最良パラメータを保存
    params_filepath = os.path.join(intermediate_results_dir, f'epoch_{epoch_counter:03d}_params.txt')
    save_results(xk, params_filepath)

    # 現在の最良相関係数を保存 (LNK_modelで更新されたグローバル変数を使用)
    save_results(-current_epoch_best_fun_value, os.path.join(intermediate_results_dir, f'epoch_{epoch_counter:03d}_correlation.txt')) # 負の値を正に戻して保存

    tqdm.write(f"--- Epoch {epoch_counter:03d} Results Saved (Correlation: {-current_epoch_best_fun_value:.4f}) at Total Runs: {total_lnk_model_runs} ---")


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


    # 状態が正常に取得できた場合のみ保存
    if A_state_final is not None:
        save_optimal_results(optimal_params, optimal_correlation_value,
                             R_state_final, A_state_final, I1_state_final, I2_state_final)
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
    # x[22] for ksi
    # x[23] for ksr

    Try_bounds = [
        (-1.0, 2.0) for _ in range(15) # alphas (L1-L15)
    ] + [
        (0.05, 1.0), # delta
        (0.1, 10.0), # a (nonlinear)
        (0.0, 5.0), # b1 (nonlinear)
        (-1.0, 0.0), # b2 (nonlinear)   
        (-0.01, 2.0), # ka (kinetic)
        (0.5, 5.0), # kfi (kinetic)
        (-0.01, 0.3), # kfr (kinetic)
        (0.001, 0.1), # ksi (kinetic)
        (0.001, 0.1)  # ksr (kinetic)
    ]
    print(f"Number of parameters in Try_bounds: {len(Try_bounds)}")
    main(Try_bounds)