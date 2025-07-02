# coding: utf-8
from tqdm import tqdm
import pprint
import math
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import differential_evolution
import yaml

import components.F_LNX as F_LNX
import components.N_LNK as N_LNK
import components.K_baccus as K_LNK

# 提供データのインプット
# cb1
# Output = np.genfromtxt("components/Provided_Data/cb1/cb1_Fourier_Result.txt")
Input = np.genfromtxt("components/Provided_Data/cb1/wn_0.0002s.txt")
# cb2
# Output = np.genfromtxt("components/Provided_Data/cb2/cb2_Fourier_Result.txt")
# Input = np.genfromtxt("components/Provided_Data/cb2/wn.txt")

# 設定ファイルの読み込み
def load_config(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# 目的関数定義
def LNK_model(x):
    #ハイパーパラメータの設定
    config_file_path = 'src/components/config/Baccus.yaml'
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
            I_start = config['I_start']
            
            tau = config['tau']
            #基底関数の数
            J = config['j']
        except KeyError as e:
            print(f"設定ファイルに必要なキーがありません: {e}")
            return np.inf
    except FileNotFoundError:
        print(f"エラー: '{config_file_path}' が見つかりません。")
    except yaml.YAMLError as exc:
        print(f"YAMLファイルのパースエラー: {exc}")
    except KeyError as e:
        print(f"設定ファイルに予期せぬキーがありません: {e}")
   
    #Linear Filterについて
    tild_Linear_Filter = np.array([])
    Linear_Filter = np.array([])
    
    #線形フィルタに通した出力結果
    tild_g = np.array([])
    g = np.array([])
    
    #線形フィルタにおけるスケーリング定数を求めるためのパラメータ
    Record1 = 0
    Record2 = 0
    
    #Nonlinearの出力結果を格納するための配列の作成
    U_Nonlinear = np.array([])
    
    #3状態を格納する配列
    R_state = np.array([])
    A_state = np.array([])
    I_state = np.array([])
    
    #評価の準備
    Result = np.array([])
    
    #Linear Filter
    for i in range(800):
        #初期化
        keep_sin_wave = 0
        for j in range(tau):
            #基底関数の計算
            keep_sin_wave += F_LNX.main(x[j-J],i*0.2,dt, tau, j + 1, J)[i]
        
    #チルダFの作成
    tild_Linear_Filter = np.append(tild_Linear_Filter, keep_sin_wave)
        
    #畳み込みでチルダgの作成
    tild_g = np.convolve(Input, tild_Linear_Filter, mode='full')
    #チルダgの作成
    #入力刺激とチルダgの分散を求める
    for i in range(80000):
        tild_g[i] = dt * tild_g[i]
        Record1 = Record1 + (tild_g[i] * tild_g[i]) * dt
        Record2 = Record2 + (Input[i] * Input[i]) * dt
    #スケーリング係数を求める
    scale_Linear = math.sqrt(Record1 / Record2)
    #スケーリング
    for i in range(800):
        keep_linear = tild_Linear_Filter[i] / scale_Linear
        Linear_Filter = np.append(Linear_Filter, keep_linear)
    
    for i in range(80000):
        keep_g = tild_g[i] / scale_Linear
        g = np.append(g, keep_g)
    
    #Nonlinearモデル
    for i in range(80000):
        #非線形性の計算
        Nonlinear_output = N_LNK.main(g[i], x[16], x[17], x[18])
        U_Nonlinear = np.append(U_Nonlinear, Nonlinear_output)
        
    #Kineticモデル
    #初期値の入力
    keep_R = R_start
    keep_A = A_start
    keep_I = I_start
    
    #4状態の計算
    R_state, A_state, I1_state, I2_state ,check= K_LNK.main(80000, U_Nonlinear, dt, keep_R, keep_A, keep_I, 0.0, x[19], x[20], x[21], 0.0, 0.0)
    #評価の準備
    Result = np.array([])

def main(Try_bounds):
    #差分進化法
    result = differential_evolution(LNK_model, Try_bounds, disp=True, updating = 'deferred', maxiter = 100, popsize = 200, strategy = 'rand1bin', workers=-1)

    #表示
    pprint.pprint(result)
    



if __name__ == "__main__":
    #探索するパラメータの範囲
    #左からlinarのシグマ15個, delta, 非線形パラメータ3個, 動的パラメータ3個
    Try_bounds = [(0.5, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (0.05, 0.2), (5, 6), (-1.5, -1), (-6, -5), (0.4, 0.6), (0.06, 0.1)]
    main(Try_bounds)

