import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.stats import spearmanr
import glob

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    import components.L_LNK as L_LNK
    import components.N_LNK as N_LNK
    import components.K_baccus as K_LNK
except ImportError:
    print("エラー: 'src/components' が見つかりません。")
    sys.exit(1)

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_data(config_path):
    """データをロードし、学習時と同じ正規化を行う"""
    cfg = load_yaml(config_path)
    base_dir = os.path.dirname(config_path)
    input_path = cfg['input_file']
    output_path = cfg['output_file']
    dt = cfg['dt']

    print(f"Loading data from: {input_path}")
    
    raw_input = np.genfromtxt(input_path)
    raw_output = np.genfromtxt(output_path)

    # 1. Input: Z-score 正規化
    input_std = np.std(raw_input)
    if input_std > 1e-9:
        norm_input = (raw_input - np.mean(raw_input)) / input_std
    else:
        norm_input = raw_input - np.mean(raw_input)

    # 2. Output: Max-Abs Scaling
    max_val = np.max(np.abs(raw_output))
    if max_val > 1e-9:
        norm_output = raw_output / max_val
    else:
        norm_output = raw_output

    return norm_input, norm_output, dt

def load_params_from_dir(results_dir):
    """
    パラメータ読み込み関数 (自動復元機能付き)
    """
    params = {}
    
    # 1. フィルタ長 J の推定
    l_files = glob.glob(os.path.join(results_dir, "L*.txt"))
    if not l_files:
        J = 50 # Lファイルがない場合の仮定値
    else:
        J = len(l_files)
    params['J'] = J

    # 2. 読み込むべきパラメータ一覧
    keys_order = ['delta', 'a', 'kappa', 'b1', 'b2', 'ka', 'kfi', 'kfr', 'ksi', 'ksr', 
                  'p_R', 'p_A', 'p_I1', 'p_I2']
    
    # 3. 欠損ファイルのチェック
    missing_files = []
    for k in keys_order:
        if not os.path.exists(os.path.join(results_dir, f"{k}.txt")):
            missing_files.append(k)

    # 4. 欠損がある場合は、epochsフォルダから最新のパラメータをロードして復元
    if missing_files or not l_files:
        print(f"警告: 以下のパラメータファイルが見つかりません: {missing_files}")
        print("epochsフォルダから最新のパラメータセットを復元します...")
        
        epoch_files = glob.glob(os.path.join(results_dir, "epochs", "epoch_*_params.txt"))
        if not epoch_files:
            print("エラー: epochsフォルダにもデータがありません。デフォルト値を使用します。")
            defaults = {'b2': 1.0, 'ka': 1.0, 'kappa': 1.0, 'a': 10.0}
            for k in keys_order:
                params[k] = defaults.get(k, 1.0)
            if 'alphas' not in params: params['alphas'] = np.zeros(J)
            return params
            
        latest_file = max(epoch_files, key=lambda x: int(os.path.basename(x).split('_')[1]))
        print(f"最新のエポックファイルを使用: {latest_file}")
        
        x = np.loadtxt(latest_file)
        
        params['alphas'] = x[0:J]
        current_idx = J
        for k in keys_order:
            if current_idx < len(x):
                params[k] = float(x[current_idx])
            else:
                params[k] = 1.0
            current_idx += 1
            
        print(f"パラメータの復元に成功しました。 (b2={params.get('b2'):.4f}, kappa={params.get('kappa'):.4f})")
        
    else:
        alphas = []
        l_files.sort(key=lambda x: int(os.path.basename(x).replace('L', '').replace('.txt', '')))
        for f in l_files:
            alphas.append(float(np.loadtxt(f)))
        params['alphas'] = np.array(alphas)
        
        for k in keys_order:
            params[k] = float(np.loadtxt(os.path.join(results_dir, f"{k}.txt")))

    # 5. ゼロ除算防止
    if params.get('b2', 0) == 0: params['b2'] = 1e-6
    if params.get('kappa', 0) == 0: params['kappa'] = 1e-6
    if params.get('ka', 0) == 0: params['ka'] = 1e-6
            
    return params

def normalize_states(p_R, p_A, p_I1, p_I2):
    total = p_R + p_A + p_I1 + p_I2
    if total > 1e-9:
        return p_R/total, p_A/total, p_I1/total, p_I2/total
    return 1.0, 0.0, 0.0, 0.0

def run_prediction(params, Input, dt, tau=1.0):
    """
    モデルを実行し、中間出力(g_t, u_t)も含めて返す
    """
    R0, A0, I10, I20 = normalize_states(params['p_R'], params['p_A'], params['p_I1'], params['p_I2'])
    t_len = len(Input)

    # 1. Linear Filter
    linear_filter_kernel, _ = L_LNK.main(params['alphas'], params['delta'], t_len, dt, tau)
    
    # 畳み込みとシフト調整
    g_full = fftconvolve(Input, linear_filter_kernel, mode='full')
    filter_len = int(tau / dt)
    shift_idx = len(linear_filter_kernel) - filter_len
    if shift_idx < 0: shift_idx = 0
    if shift_idx >= len(g_full): shift_idx = 0
    
    g_t = g_full[shift_idx : shift_idx + t_len]
    
    # 2. Nonlinear Module (u_t)
    u_t = N_LNK.main(
        g_t, 
        params['a'], 
        params['kappa'], 
        params['b1'], 
        params['b2'], 
        params['ka']
    )
    
    # 3. Kinetic Model (A)
    R, A, I1, I2, check = K_LNK.main(
        len(u_t), u_t, dt, 
        R0, A0, I10, I20,
        params['ka'], params['kfi'], params['kfr'], params['ksi'], params['ksr'],
        label="Validation"
    )
    
    return g_t, u_t, A, check

def main():
    # ================= 設定エリア =================
    # DATA_CONFIG_PATH = "config/data/ret2p-1.yaml"
    DATA_CONFIG_PATH = "config/data/Ucb2.yaml"
    RESULTS_DIR = "scripts/20251208_00" 
    TAU = 1.0 
    # ============================================

    if len(sys.argv) > 1:
        RESULTS_DIR = sys.argv[1]
    
    if not os.path.exists(RESULTS_DIR):
        print(f"エラー: ディレクトリが見つかりません: {RESULTS_DIR}")
        return

    # データロード
    Input, Output_exp, dt = load_data(DATA_CONFIG_PATH)
    
    # パラメータロード
    print(f"Loading parameters from {RESULTS_DIR}...")
    params = load_params_from_dir(RESULTS_DIR)

    # 長さ合わせ
    min_len = min(len(Input), len(Output_exp))
    Input = Input[:min_len]
    Output_exp = Output_exp[:min_len]

    # 予測実行 (中間出力も取得)
    print("Running model prediction...")
    g_t, u_t, Prediction, check = run_prediction(params, Input, dt, tau=TAU)
    
    # 結果の整形
    if check != 1:
        print("\n!!!!!!!! 警告 !!!!!!!!")
        print("Kineticモデルの計算が失敗しました。予測値はゼロ埋めされます。")
        Prediction = np.zeros_like(Output_exp)
        corr = 0.0
    
    # 最終的な長さ合わせ (計算過程でずれる可能性があるため)
    min_len_eval = min(len(Output_exp), len(Prediction), len(g_t), len(u_t))
    
    Output_exp = Output_exp[:min_len_eval]
    Prediction = Prediction[:min_len_eval]
    g_t = g_t[:min_len_eval]
    u_t = u_t[:min_len_eval]
    Input_plot = Input[:min_len_eval]

    if check == 1:
        corr, _ = spearmanr(Output_exp, Prediction)
        print(f"Spearman Correlation: {corr:.6f}")

    # ================= 保存処理 (追加箇所) =================
    save_dir = os.path.join(RESULTS_DIR, "validation")
    os.makedirs(save_dir, exist_ok=True)

    print("時系列データを保存中...")
    np.savetxt(os.path.join(save_dir, "g.txt"), g_t, fmt='%.6f')
    np.savetxt(os.path.join(save_dir, "u.txt"), u_t, fmt='%.6f')
    np.savetxt(os.path.join(save_dir, "predict.txt"), Prediction, fmt='%.6f')
    print(f"保存完了: {save_dir}/{{g.txt, u.txt, predict.txt}}")
    # ====================================================

    # プロット作成 (4段構成)
    total_time = len(Input_plot) * dt
    time_axis = np.arange(len(Input_plot)) * dt
    major_ticks = np.arange(0, int(total_time) + 1, 1)

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=False) 
    
    def setup_axis(ax, title, ylabel):
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, total_time)
        ax.set_xticks(major_ticks) 
        ax.grid(which='major', alpha=0.5, linestyle='-') 
        ax.tick_params(axis='x', labelbottom=True) 
        ax.set_xlabel('Time (s)', fontsize=10) 

    # 1. Input
    axes[0].plot(time_axis, Input_plot, color='gray', linewidth=1)
    setup_axis(axes[0], '1. Input Stimulus (Normalized)', 'Contrast')

    # 2. Linear Filter Output (g_t)
    axes[1].plot(time_axis, g_t, color='blue', linewidth=1)
    setup_axis(axes[1], '2. Linear Filter Output g(t)', 'Filtered Signal')

    # 3. Nonlinear Output (u_t)
    axes[2].plot(time_axis, u_t, color='green', linewidth=1)
    setup_axis(axes[2], '3. Nonlinear Output u(t) (Scaled Rate)', 'Rate (u)')

    # 4. Final Response
    axes[3].plot(time_axis, Output_exp, color='black', alpha=0.5, label='Experiment', linewidth=1.5)
    axes[3].plot(time_axis, Prediction, color='red', alpha=0.8, label=f'Model (Corr={corr:.3f})', linewidth=1.5)
    setup_axis(axes[3], '4. Final Response (Prediction)', 'Response')
    axes[3].legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "detailed_analysis.png")
    plt.savefig(plot_path)
    print(f"詳細グラフを保存しました: {plot_path}")

if __name__ == "__main__":
    main()