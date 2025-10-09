# plot_specific_params.py
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import yaml
import time
from scipy.stats import spearmanr

# components 以下のモジュールをインポートするためにsys.pathに追加
import sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # このファイル(__file__)の絶対パスのルートディレクトリを取得して一個上の階層を指定
sys.path.append(project_root_dir)
# これを追加することで、 以下のモジュールのインポートを簡単にする
import components.F_LNK as F_LNK
import components.N_LNK as N_LNK
import components.K_baccus as K_LNK

# 設定ファイルを読み込む関数
def load_config(filepath):
    """
    ハイパーパラメータ設定ファイルを読み込みます。
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# config.yaml のパス
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(script_dir, "components", "config", "Baccus.yaml")

# 設定の読み込み
try:
    config = load_config(config_file_path)
except FileNotFoundError:
    print(f"エラー: '{config_file_path}' が見つかりません。設定ファイルを読み込めません。")
    sys.exit(1)
except yaml.YAMLError as exc:
    print(f"YAMLファイルのパースエラー: {exc}")
    sys.exit(1)
except KeyError as e:
    print(f"設定ファイルに予期せぬキーがありません: {e}")
    sys.exit(1)

# --- ここで日付を手動で指定します (グラフ保存先のディレクトリ名に使用) ---
date_str = "20250711_12" # 任意のディレクトリ名に設定してください

# 設定から必要な値を取得
data_options = config['data_options']
J = config['J'] # 基底関数の数

# Provided_Data をロード
if data_options == "cb1":
    print("cb1のデータを使用します")
    Input_data = np.genfromtxt(os.path.join(script_dir, "components", "Provided_Data", "cb1", "wn_0.0002s.txt"))
    Output_data = np.genfromtxt(os.path.join(script_dir, "components", "Provided_Data", "cb1", "cb1_Fourier_Result.txt"))
elif data_options == "cb2":
    print("cb2のデータを使用します")
    Input_data = np.genfromtxt(os.path.join(script_dir, "components", "Provided_Data", "cb2", "wn.txt"))
    Output_data = np.genfromtxt(os.path.join(script_dir, "components", "Provided_Data", "cb2", "cb2_Fourier_Result.txt"))
else:
    print(f"エラー: 未知のデータオプション '{data_options}' です。入出力データを決定できません。")
    sys.exit(1)

# LNK_model 関数の再定義 (グラフ描画のために必要な部分のみ)
def LNK_model_for_plot(x, Input_data_arg, Output_data_arg, dt, J, config_params):
    """
    最適なパラメータを用いてLNKモデルを実行し、A_stateと相関を返す。
    """
    R_start = config_params['R_start']
    A_start = config_params['A_start']
    I1_start = config_params['I_start']
    I2_start = 0.0
    tau = config_params['tau']

    alphas = x[0:J]
    delta = x[J]
    a_nonlinear = x[J+1]
    b1_nonlinear = x[J+2]
    b2_nonlinear = x[J+3]
    ka_kinetic = x[J+4]
    kfi_kinetic = x[J+5]
    kfr_kinetic = x[J+6]

    ksi_kinetic = 0.0
    ksr_kinetic = 0.0

    Linear_Filter_kernel, _ = F_LNK.main(alphas, delta, dt, tau, J)
    tild_g_full = np.convolve(Input_data_arg, Linear_Filter_kernel, mode='full')
    g_len = len(Input_data_arg)
    tild_g = tild_g_full[:g_len]

    Record1 = np.sum(tild_g * tild_g) * dt
    Record2 = np.sum(Input_data_arg * Input_data_arg) * dt
    
    if Record2 <= 1e-9:
        scale_Linear = 1.0
    else:
        scale_Linear = np.sqrt(Record1 / Record2)

    g = tild_g / scale_Linear

    U_Nonlinear = N_LNK.main(g, a_nonlinear, b1_nonlinear, b2_nonlinear)

    R_state, A_state, I1_state, I2_state, check_raw = K_LNK.main(
        len(U_Nonlinear), U_Nonlinear, dt, R_start, A_start, I1_start, I2_start,
        ka_kinetic, kfi_kinetic, kfr_kinetic, ksi_kinetic, ksr_kinetic,
        label="Plotting Run"
    )
    
    check = None
    if isinstance(check_raw, np.ndarray):
        if check_raw.size == 1:
            check = check_raw.item()
        else:
            check = 0 # 複数要素の配列は失敗とみなす
    else:
        check = check_raw

    if check != 1:
        print("Warning: Kinetic model failed in LNK_model_for_plot. Returning None for states.")
        return None, None, None, None, 0.0

    keep_Post = (-1) * A_state[:len(Output_data_arg)]
    min_len_corr = min(len(Output_data_arg), len(keep_Post))
    Output_trimmed_corr = Output_data_arg[:min_len_corr]
    keep_Post_trimmed_corr = keep_Post[:min_len_corr]

    if min_len_corr > 1:
        correlation, _ = spearmanr(Output_trimmed_corr, keep_Post_trimmed_corr)
    else:
        correlation = 0.0

    return R_state, A_state, I1_state, I2_state, correlation


# 結果を保存するディレクトリを準備
results_base_dir = os.path.join(project_root_dir, 'results', 'Baccus_' + data_options)
target_result_dir = os.path.join(results_base_dir, date_str)
os.makedirs(target_result_dir, exist_ok=True) # ディレクトリが存在しない場合は作成

# --- 比較する2つのパラメータセット ---
# 94-parents.txt のパラメータ (以前のModel 1)
params_94_parents = np.array([
    0.992053, 0.057222, 0.214281, 0.240299, 0.425180,
    0.385704, 0.017517, 0.696799, 0.247215, 0.175373,
    0.424967, 0.153786, 0.064951, 0.451844, 0.585853,
    0.180307, 4.209184, 0.665612, -0.281921, 0.625779,
    0.948792, 0.221268
])

# 03-params.txt のパラメータ (以前のModel 2)
params_03_params = np.array([
    0.850741, 0.043609, 0.314961, 0.115735, 0.405859,
    0.181998, 0.488759, 0.201215, 0.388303, 0.646640,
    0.021255, 0.884164, 0.723802, 0.443743, 0.024795,
    0.083580, 9.336384, 4.152542, -0.673780, 0.419078,
    0.164253, 0.170182
])

print("提供されたパラメータデータを用いてLNKモデルの出力を生成中...")

try:
    # --- 94-parents.txt の出力を計算 ---
    print("94-parents.txt の出力を計算中...")
    _, calculated_A_state_94, _, _, final_correlation_94 = LNK_model_for_plot(
        params_94_parents, Input_data, Output_data, config['dt'], J, config
    )
    if calculated_A_state_94 is None:
        print("エラー: 94-parents.txt のLNKモデル実行が失敗しました。グラフを作成できません。")
        sys.exit(1)
    calculated_output_trimmed_94 = (-1) * calculated_A_state_94[:len(Output_data)]

    # --- 03-params.txt の出力を計算 ---
    print("03-params.txt の出力を計算中...")
    _, calculated_A_state_03, _, _, final_correlation_03 = LNK_model_for_plot(
        params_03_params, Input_data, Output_data, config['dt'], J, config
    )
    if calculated_A_state_03 is None:
        print("エラー: 03-params.txt のLNKモデル実行が失敗しました。グラフを作成できません。")
        sys.exit(1)
    calculated_output_trimmed_03 = (-1) * calculated_A_state_03[:len(Output_data)]

    # 時間軸の生成 (共通の長さに合わせる)
    min_overall_len = min(len(calculated_output_trimmed_94), len(calculated_output_trimmed_03), len(Output_data))
    time_axis = np.arange(0, min_overall_len * config['dt'], config['dt'])

    # 共通の長さにトリミング
    Provided_Output_trimmed = Output_data[:min_overall_len]
    calculated_output_trimmed_94 = calculated_output_trimmed_94[:min_overall_len]
    calculated_output_trimmed_03 = calculated_output_trimmed_03[:min_overall_len]


    # --- 共通のデータ抽出と処理 (時間軸1000秒以上) ---
    time_indices_1000_plus = np.where(time_axis >= 1000)[0]
    
    if len(time_indices_1000_plus) == 0:
        print("警告: 時間軸1000秒以上のデータポイントが見つかりませんでした。関連するグラフは作成されません。")
    else:
        start_idx_1000_plus = time_indices_1000_plus[0]

        time_axis_1000_plus = time_axis[start_idx_1000_plus:]
        Provided_Output_1000_plus = Provided_Output_trimmed[start_idx_1000_plus:]
        calculated_output_1000_plus_94 = calculated_output_trimmed_94[start_idx_1000_plus:]
        calculated_output_1000_plus_03 = calculated_output_trimmed_03[start_idx_1000_plus:]


        # --- 処理済みデータセットの準備 ---

        # 平均0移動 & L2ノルム正規化されたデータ
        mean_centered_Provided_Output_L2 = Provided_Output_1000_plus.copy()
        mean_centered_calculated_Output_L2_94 = calculated_output_1000_plus_94.copy()
        mean_centered_calculated_Output_L2_03 = calculated_output_1000_plus_03.copy()

        if len(mean_centered_Provided_Output_L2) > 0:
            mean_centered_Provided_Output_L2 -= np.mean(mean_centered_Provided_Output_L2)
        if len(mean_centered_calculated_Output_L2_94) > 0:
            mean_centered_calculated_Output_L2_94 -= np.mean(mean_centered_calculated_Output_L2_94)
        if len(mean_centered_calculated_Output_L2_03) > 0:
            mean_centered_calculated_Output_L2_03 -= np.mean(mean_centered_calculated_Output_L2_03)
        
        norm_provided_mc = np.linalg.norm(mean_centered_Provided_Output_L2)
        norm_calculated_mc_94 = np.linalg.norm(mean_centered_calculated_Output_L2_94)
        norm_calculated_mc_03 = np.linalg.norm(mean_centered_calculated_Output_L2_03)

        if norm_provided_mc != 0:
            mean_centered_Provided_Output_L2 /= norm_provided_mc
        if norm_calculated_mc_94 != 0:
            mean_centered_calculated_Output_L2_94 /= norm_calculated_mc_94
        if norm_calculated_mc_03 != 0:
            mean_centered_calculated_Output_L2_03 /= norm_calculated_mc_03

        # L2ノルム正規化のみされたデータ (平均0移動なし)
        l2_normalized_Provided_Output = Provided_Output_1000_plus.copy()
        l2_normalized_calculated_Output_94 = calculated_output_1000_plus_94.copy()
        l2_normalized_calculated_Output_03 = calculated_output_1000_plus_03.copy()

        norm_provided_l2 = np.linalg.norm(l2_normalized_Provided_Output)
        norm_calculated_l2_94 = np.linalg.norm(l2_normalized_calculated_Output_94)
        norm_calculated_l2_03 = np.linalg.norm(l2_normalized_calculated_Output_03)

        if norm_provided_l2 != 0:
            l2_normalized_Provided_Output /= norm_provided_l2
        if norm_calculated_l2_94 != 0:
            l2_normalized_calculated_Output_94 /= norm_calculated_l2_94
        if norm_calculated_l2_03 != 0:
            l2_normalized_calculated_Output_03 /= norm_calculated_l2_03


        # --- 既存のグラフ: 時間軸1000以上全体 & 平均0移動 & L2ノルム正規化 (単一グラフ) ---
        plt.figure(figsize=(15, 6))
        plt.suptitle(f'Normalized & Mean-Centered LNK Model Comparison (Data: {data_options}) [Time Axis >= 1000s, Mean Shifted to 0, L2-Normalized]', fontsize=16)

        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(time_axis_1000_plus, mean_centered_Provided_Output_L2, 
                 label='Normalized & Mean-Centered Provided Output', alpha=0.7, color='blue') 
        ax1.plot(time_axis_1000_plus, mean_centered_calculated_Output_L2_94, 
                 label='Normalized & Mean-Centered LNK Model (94-parents.txt)', linestyle='--', alpha=0.7, color='orange')
        ax1.plot(time_axis_1000_plus, mean_centered_calculated_Output_L2_03, 
                 label='Normalized & Mean-Centered LNK Model (03-params.txt)', linestyle=':', alpha=0.7, color='red')
        
        ax1.set_title('Normalized & Mean-Centered Outputs - Time >= 1000s')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Normalized Amplitude (L2-norm = 1)')
        ax1.legend()
        ax1.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        graph_filepath_1000_plus_mean_centered_l2_norm = os.path.join(target_result_dir, 'LNK_model_output_comparison_time_1000_plus_mean_centered_l2_norm.png')
        plt.savefig(graph_filepath_1000_plus_mean_centered_l2_norm)
        print(f"時間軸1000秒以上で平均0移動＆L2正規化グラフを {graph_filepath_1000_plus_mean_centered_l2_norm} に保存しました。")


        # --- 既存のグラフ: 時間軸1000秒〜2000秒に拡大 (正規化＆平均0移動済み) ---
        plt.figure(figsize=(15, 6))
        plt.suptitle(f'Normalized & Mean-Centered LNK Model Comparison (Data: {data_options}) [Time Axis 1000s-2000s, Mean Shifted to 0, L2-Normalized]', fontsize=16)

        zoom_start_1 = 1000
        zoom_end_1 = 2000
        zoom_indices_1 = np.where((time_axis_1000_plus >= zoom_start_1) & (time_axis_1000_plus <= zoom_end_1))[0]

        if len(zoom_indices_1) == 0:
            print(f"警告: 時間軸 {zoom_start_1}s-{zoom_end_1}s のデータポイントが見つかりませんでした。拡大グラフは作成されません。")
        else:
            ax2 = plt.subplot(1, 1, 1)
            ax2.plot(time_axis_1000_plus[zoom_indices_1], mean_centered_Provided_Output_L2[zoom_indices_1], 
                     label='Normalized & Mean-Centered Provided Output', alpha=0.7, color='blue') 
            ax2.plot(time_axis_1000_plus[zoom_indices_1], mean_centered_calculated_Output_L2_94[zoom_indices_1], 
                     label='Normalized & Mean-Centered LNK Model (94-parents.txt)', linestyle='--', alpha=0.7, color='orange')
            ax2.plot(time_axis_1000_plus[zoom_indices_1], mean_centered_calculated_Output_L2_03[zoom_indices_1], 
                     label='Normalized & Mean-Centered LNK Model (03-params.txt)', linestyle=':', alpha=0.7, color='red')
            
            ax2.set_title(f'Normalized & Mean-Centered Outputs - Time {zoom_start_1}s-{zoom_end_1}s')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Normalized Amplitude (L2-norm = 1)')
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlim(zoom_start_1, zoom_end_1)

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            graph_filepath_zoom_1 = os.path.join(target_result_dir, f'LNK_model_output_comparison_time_{zoom_start_1}s_to_{zoom_end_1}s_normalized_mean_centered_l2_norm.png')
            plt.savefig(graph_filepath_zoom_1)
            print(f"時間軸{zoom_start_1}s-{zoom_end_1}sの拡大グラフを {graph_filepath_zoom_1} に保存しました。")


        # --- 既存のグラフ: 時間軸2000秒〜3000秒に拡大 (正規化＆平均0移動済み) ---
        plt.figure(figsize=(15, 6))
        plt.suptitle(f'Normalized & Mean-Centered LNK Model Comparison (Data: {data_options}) [Time Axis 2000s-3000s, Mean Shifted to 0, L2-Normalized]', fontsize=16)

        zoom_start_2 = 2000
        zoom_end_2 = 3000
        zoom_indices_2 = np.where((time_axis_1000_plus >= zoom_start_2) & (time_axis_1000_plus <= zoom_end_2))[0]

        if len(zoom_indices_2) == 0:
            print(f"警告: 時間軸 {zoom_start_2}s-{zoom_end_2}s のデータポイントが見つかりませんでした。拡大グラフは作成されません。")
        else:
            ax3 = plt.subplot(1, 1, 1)
            ax3.plot(time_axis_1000_plus[zoom_indices_2], mean_centered_Provided_Output_L2[zoom_indices_2], 
                     label='Normalized & Mean-Centered Provided Output', alpha=0.7, color='blue') 
            ax3.plot(time_axis_1000_plus[zoom_indices_2], mean_centered_calculated_Output_L2_94[zoom_indices_2], 
                     label='Normalized & Mean-Centered LNK Model (94-parents.txt)', linestyle='--', alpha=0.7, color='orange')
            ax3.plot(time_axis_1000_plus[zoom_indices_2], mean_centered_calculated_Output_L2_03[zoom_indices_2], 
                     label='Normalized & Mean-Centered LNK Model (03-params.txt)', linestyle=':', alpha=0.7, color='red')
            
            ax3.set_title(f'Normalized & Mean-Centered Outputs - Time {zoom_start_2}s-{zoom_end_2}s')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Normalized Amplitude (L2-norm = 1)')
            ax3.legend()
            ax3.grid(True)
            ax3.set_xlim(zoom_start_2, zoom_end_2)

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            graph_filepath_zoom_2 = os.path.join(target_result_dir, f'LNK_model_output_comparison_time_{zoom_start_2}s_to_{zoom_end_2}s_normalized_mean_centered_l2_norm.png')
            plt.savefig(graph_filepath_zoom_2)
            print(f"時間軸{zoom_start_2}s-{zoom_end_2}sの拡大グラフを {graph_filepath_zoom_2} に保存しました。")


        # --- 新規グラフ: 時間軸1000以上のみ & L2ノルム正規化 (平均0移動なし) ---
        plt.figure(figsize=(15, 6))
        plt.suptitle(f'L2-Normalized LNK Model Comparison (Data: {data_options}) [Time Axis >= 1000s, L2-Normalized Only]', fontsize=16)

        ax4 = plt.subplot(1, 1, 1)
        ax4.plot(time_axis_1000_plus, l2_normalized_Provided_Output, 
                 label='L2-Normalized Provided Output', alpha=0.7, color='blue') 
        ax4.plot(time_axis_1000_plus, l2_normalized_calculated_Output_94, 
                 label='L2-Normalized LNK Model (94-parents.txt)', linestyle='--', alpha=0.7, color='orange')
        ax4.plot(time_axis_1000_plus, l2_normalized_calculated_Output_03, # Model 2 を赤色で追加
                 label='L2-Normalized LNK Model (03-params.txt)', linestyle=':', alpha=0.7, color='red')
        
        ax4.set_title('L2-Normalized Outputs - Time >= 1000s (Not Mean-Centered)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Normalized Amplitude (L2-norm = 1)')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        graph_filepath_1000_plus_l2_norm_only = os.path.join(target_result_dir, 'LNK_model_output_comparison_time_1000_plus_l2_norm_only.png')
        plt.savefig(graph_filepath_1000_plus_l2_norm_only)
        print(f"時間軸1000秒以上でL2正規化のみのグラフを {graph_filepath_1000_plus_l2_norm_only} に保存しました。")


        # --- 新規グラフ: 時間軸1000秒〜2000秒に拡大 (L2ノルム正規化のみ) ---
        plt.figure(figsize=(15, 6))
        plt.suptitle(f'L2-Normalized LNK Model Output Comparison (Data: {data_options}) [Time Axis 1000s-2000s, L2-Normalized Only]', fontsize=16)

        if len(zoom_indices_1) == 0: # zoom_indices_1は既に計算済み
            print(f"警告: 時間軸 {zoom_start_1}s-{zoom_end_1}s のデータポイントが見つからなかったため、L2正規化のみの拡大グラフは作成されません。")
        else:
            ax5 = plt.subplot(1, 1, 1)
            ax5.plot(time_axis_1000_plus[zoom_indices_1], l2_normalized_Provided_Output[zoom_indices_1], 
                     label='L2-Normalized Provided Output', alpha=0.7, color='blue') 
            ax5.plot(time_axis_1000_plus[zoom_indices_1], l2_normalized_calculated_Output_94[zoom_indices_1], 
                     label='L2-Normalized LNK Model (94-parents.txt)', linestyle='--', alpha=0.7, color='orange')
            ax5.plot(time_axis_1000_plus[zoom_indices_1], l2_normalized_calculated_Output_03[zoom_indices_1], # Model 2 を赤色で追加
                     label='L2-Normalized LNK Model (03-params.txt)', linestyle=':', alpha=0.7, color='red')
            
            ax5.set_title(f'L2-Normalized Outputs - Time {zoom_start_1}s-{zoom_end_1}s (Not Mean-Centered)')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Normalized Amplitude (L2-norm = 1)')
            ax5.legend()
            ax5.grid(True)
            ax5.set_xlim(zoom_start_1, zoom_end_1)

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            graph_filepath_zoom_1_l2_only = os.path.join(target_result_dir, f'LNK_model_output_comparison_time_{zoom_start_1}s_to_{zoom_end_1}s_l2_norm_only.png')
            plt.savefig(graph_filepath_zoom_1_l2_only)
            print(f"時間軸{zoom_start_1}s-{zoom_end_1}sのL2正規化のみの拡大グラフを {graph_filepath_zoom_1_l2_only} に保存しました。")


        # --- 新規グラフ: 時間軸2000秒〜3000秒に拡大 (L2ノルム正規化のみ) ---
        plt.figure(figsize=(15, 6))
        plt.suptitle(f'L2-Normalized LNK Model Output Comparison (Data: {data_options}) [Time Axis 2000s-3000s, L2-Normalized Only]', fontsize=16)

        if len(zoom_indices_2) == 0: # zoom_indices_2は既に計算済み
            print(f"警告: 時間軸 {zoom_start_2}s-{zoom_end_2}s のデータポイントが見つからなかったため、L2正規化のみの拡大グラフは作成されません。")
        else:
            ax6 = plt.subplot(1, 1, 1)
            ax6.plot(time_axis_1000_plus[zoom_indices_2], l2_normalized_Provided_Output[zoom_indices_2], 
                     label='L2-Normalized Provided Output', alpha=0.7, color='blue') 
            ax6.plot(time_axis_1000_plus[zoom_indices_2], l2_normalized_calculated_Output_94[zoom_indices_2], 
                     label='L2-Normalized LNK Model (94-parents.txt)', linestyle='--', alpha=0.7, color='orange')
            ax6.plot(time_axis_1000_plus[zoom_indices_2], l2_normalized_calculated_Output_03[zoom_indices_2], # Model 2 を赤色で追加
                     label='L2-Normalized LNK Model (03-params.txt)', linestyle=':', alpha=0.7, color='red')
            
            ax6.set_title(f'L2-Normalized Outputs - Time {zoom_start_2}s-{zoom_end_2}s (Not Mean-Centered)')
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Normalized Amplitude (L2-norm = 1)')
            ax6.legend()
            ax6.grid(True)
            ax6.set_xlim(zoom_start_2, zoom_end_2)

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            graph_filepath_zoom_2_l2_only = os.path.join(target_result_dir, f'LNK_model_output_comparison_time_{zoom_start_2}s_to_{zoom_end_2}s_l2_norm_only.png')
            plt.savefig(graph_filepath_zoom_2_l2_only)
            print(f"時間軸{zoom_start_2}s-{zoom_end_2}sのL2正規化のみの拡大グラフを {graph_filepath_zoom_2_l2_only} に保存しました。")


except Exception as e:
    print(f"グラフ作成中に予期せぬエラーが発生しました: {e}")
    print("LNK_model_for_plotが正しく動作しているか確認してください。")

print("\nグラフ作成スクリプトが完了しました。")