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

# 提供データのインプット
data_options =  "cb1"
# data_options =  "cb2"
if data_options == "cb1":
    print("cb1のデータを使用します")
    #cb1データ
    Input = np.genfromtxt("/app/src/components/Provided_Data/cb1/wn_0.0002s.txt")
    Output = np.genfromtxt("/app/src/components/Provided_Data/cb1/cb1_Fourier_Result.txt")
elif data_options == "cb2":
    print("cb2のデータを使用します")
    #cb2データ
    Input = np.genfromtxt("/app/src/components/Provided_Data/cb2/wn.txt")
    Output = np.genfromtxt("/app/src/components/Provided_Data/cb2/cb2_Fourier_Result.txt")

# 設定ファイルの読み込み
def load_config(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# 実験結果の保存
def save_results(result, filepath):
    print(f"結果を保存: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a', encoding='utf-8') as file:
        file.write(str(result) + '\n')
# 目的関数定義
def LNK_model(x):
    global total_lnk_model_runs
    global failed_lnk_model_runs
    global date_str
    global data_options
    total_lnk_model_runs += 1 # 関数の開始時に合計実行回数をインクリメント
    #ハイパーパラメータの設定
    config_file_path = '/app/src/components/config/Baccus.yaml'
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
    print("線形フィルターの計算を開始します...")
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
    print("非線形モデルの計算を開始します...")
    U_Nonlinear = np.array([N_LNK.main(val, a_nonlinear, b1_nonlinear, b2_nonlinear) for val in tqdm(g)])
        
    #Kineticモデル
    # K_LNK.mainの引数を修正: time_steps, u_input, dt, R_start, A_start, I1_start, I2_start, ka, kfi, kfr, ksi, ksr
    print("Kineticモデルの計算を開始します...")
    R_state, A_state, I1_state, I2_state ,check= K_LNK.main(len(U_Nonlinear), U_Nonlinear, dt, R_start, A_start, I1_start, I2_start, ka_kinetic, kfi_kinetic, kfr_kinetic, ksi_kinetic, ksr_kinetic)
    
    print("スピアマンの相関係数を計算します...")
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
        
        # パラメータの保存
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(script_dir) 
        results_base_dir = os.path.join(project_root_dir, 'results', 'Baccus'+ data_options)

        # パラメータ用のディレクトリ
        param_results_dir = os.path.join(results_base_dir, date_str)
        os.makedirs(param_results_dir, exist_ok=True)
        
        for i in range(J):#線形フィルタのパラメータを保存
            save_results(x[i], os.path.join(param_results_dir, f'L{i+1}.txt'))
        save_results(x[J], os.path.join(param_results_dir, 'delta.txt'))
        save_results(x[J+1], os.path.join(param_results_dir, 'a.txt'))
        save_results(x[J+2], os.path.join(param_results_dir, 'b1.txt'))
        save_results(x[J+3], os.path.join(param_results_dir, 'b2.txt'))
        save_results(x[J+4], os.path.join(param_results_dir, 'ka.txt'))
        save_results(x[J+5], os.path.join(param_results_dir, 'kfi.txt'))
        save_results(x[J+6], os.path.join(param_results_dir, 'kfr.txt'))
        save_results(correlation, os.path.join(param_results_dir, 'correlation.txt'))
        
        # 状態の保存
        state_results_dir = os.path.join(results_base_dir, 'state', date_str) 
        os.makedirs(state_results_dir, exist_ok=True)
        save_results(R_state, os.path.join(state_results_dir, 'R_state.txt'))
        save_results(A_state, os.path.join(state_results_dir, 'A_state.txt'))
        save_results(I1_state, os.path.join(state_results_dir, 'I1_state.txt'))
        save_results(I2_state, os.path.join(state_results_dir, 'I2_state.txt'))

        # 最小化問題のため相関の負の値を返す
        correlation = (-1) * correlation 
    else:
        print("Kinetic model が失敗しました.")
        failed_lnk_model_runs += 1 # Kineticモデルが失敗した場合は失敗としてマーク
        correlation = 1000.0 # 状態が不正な場合は大きなペナルティ
    # 結果の表示
    print(f"相関係数: {correlation:.4f}")
    if total_lnk_model_runs > 0:
        failure_rate = (failed_lnk_model_runs / total_lnk_model_runs) * 100
        print(f"LNK_model の失敗回数/合計実行回数(失敗率): {failed_lnk_model_runs}/{total_lnk_model_runs} ({failure_rate:.2f}%)")
    else:
        print("実行記録がありません。")
    return correlation

def main(Try_bounds):
    #差分進化法
    # disp=True で進捗を表示
    # updating='deferred' で更新を遅延させる
    # maxiter=100 で最大反復回数を設定
    # popsize=200 で個体群のサイズを設定
    # strategy='rand1bin' で戦略を設定
    # workers=-1 で全てのCPUコアを使用
    # 差分進化アルゴリズムの目的関数呼び出し: 約 20,000 回 (maxiter * popsize)
    result = differential_evolution(LNK_model, Try_bounds, disp=True, updating = 'deferred', maxiter = 100, popsize = 200, strategy = 'rand1bin', workers=-1)

    #表示
    pprint.pprint(result)
    # パラメータの保存
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(script_dir) 
    results_base_dir = os.path.join(project_root_dir, 'results', 'Baccus_'+ data_options)

    # パラメータ用のディレクトリ
    param_results_dir = os.path.join(results_base_dir, date_str)
    os.makedirs(param_results_dir, exist_ok=True)
    save_results(result.x, os.path.join(param_results_dir, 'result.txt'))

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