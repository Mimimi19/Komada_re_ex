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
    input_path = cfg['input_file']
    output_path = cfg['output_file']
    dt = cfg['dt']

    print(f"Loading data from: {input_path}")
    raw_input = np.genfromtxt(input_path)
    raw_output = np.genfromtxt(output_path)

    # Input: Z-score 正規化
    input_std = np.std(raw_input)
    if input_std > 1e-9:
        norm_input = (raw_input - np.mean(raw_input)) / input_std
    else:
        norm_input = raw_input - np.mean(raw_input)

    # Output: Max-Abs Scaling
    max_val = np.max(np.abs(raw_output))
    if max_val > 1e-9:
        norm_output = raw_output / max_val
    else:
        norm_output = raw_output

    return norm_input, norm_output, dt

def load_params_from_dir(results_dir):
    """パラメータ読み込み"""
    params = {}
    l_files = glob.glob(os.path.join(results_dir, "L*.txt"))
    alphas = []
    # L1, L2... の順にソート
    l_files.sort(key=lambda x: int(os.path.basename(x).replace('L', '').replace('.txt', '')))
    
    for f in l_files:
        val = np.loadtxt(f)
        alphas.append(float(val))
    
    params['alphas'] = np.array(alphas)
    params['J'] = len(alphas)

    keys = ['delta', 'a', 'kappa', 'b1', 'b2', 'ka', 'kfi', 'kfr', 'ksi', 'ksr', 
            'p_R', 'p_A', 'p_I1', 'p_I2']
    
    for k in keys:
        path = os.path.join(results_dir, f"{k}.txt")
        if os.path.exists(path):
            params[k] = float(np.loadtxt(path))
        else:
            params[k] = 0.0
    return params

def normalize_states(p_R, p_A, p_I1, p_I2):
    total = p_R + p_A + p_I1 + p_I2
    if total > 1e-9:
        return p_R/total, p_A/total, p_I1/total, p_I2/total
    return 1.0, 0.0, 0.0, 0.0

def run_prediction(params, Input, dt, tau=1.0):
    """
    モデルを実行し、中間出力(g_t, u_t)も含めて返す
    【重要】BaccusModel.py と同じロジックで計算する
    """
    R0, A0, I10, I20 = normalize_states(params['p_R'], params['p_A'], params['p_I1'], params['p_I2'])
    
    # 1. Linear Filter カーネル生成 (長さを制限)
    filter_points = int(tau / dt) + 1
    linear_filter_kernel, _ = L_LNK.main(params['alphas'], params['delta'], filter_points, dt, tau)
    
    # --- 修正箇所: mode='full' で畳み込み & シフト処理 ---
    g_full = fftconvolve(Input, linear_filter_kernel, mode='full')
    
    # シフト量
    shift_idx = int(tau / dt) 
    
    # シフトして切り出し (入力と同じ長さに合わせる)
    if len(g_full) > shift_idx + len(Input):
        g_t = g_full[shift_idx : shift_idx + len(Input)]
    else:
        g_t = g_full[:len(Input)]
    
    # --- 修正箇所: 強制正規化 ---
    g_std = np.std(g_t)
    if g_std > 1e-9:
        g_t = g_t / g_std

    # 2. Nonlinear Module
    u_t = N_LNK.main(
        g_t, 
        params['a'], 
        params['kappa'], 
        params['b1'], 
        params['b2'], 
        params['ka']
    )
    
    # 3. Kinetic Model
    R, A, I1, I2, check = K_LNK.main(
        len(u_t), u_t, dt, 
        R0, A0, I10, I20,
        params['ka'], params['kfi'], params['kfr'], params['ksi'], params['ksr'],
        label="Validation"
    )
    
    return g_t, u_t, A, check

def main():
    # ================= 設定エリア =================
    # DATA_CONFIG_PATH = "config/data/Ucb2.yaml"
    DATA_CONFIG_PATH = "config/data/ret2p-1.yaml"
    # 最新の結果ディレクトリを指定してください
    RESULTS_DIR = "scripts/results/Baccus_cb1/20251212_12" 
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
    try:
        params = load_params_from_dir(RESULTS_DIR)
    except Exception as e:
        print(f"パラメータ読み込みエラー: {e}")
        return

    # 長さ合わせ
    min_len = min(len(Input), len(Output_exp))
    Input = Input[:min_len]
    Output_exp = Output_exp[:min_len]

    # 予測実行
    print("Running model prediction...")
    g_t, u_t, Prediction, check = run_prediction(params, Input, dt, tau=TAU)
    
    # 相関計算 (マスク適用)
    corr = 0.0
    if check == 1:
        # 長さ調整
        min_len_eval = min(len(Output_exp), len(Prediction))
        Output_exp = Output_exp[:min_len_eval]
        Prediction = Prediction[:min_len_eval]
        g_t = g_t[:min_len_eval]
        u_t = u_t[:min_len_eval]
        Input_plot = Input[:min_len_eval]

        # 最初の1秒を無視して相関計算
        mask_seconds = 1.0
        mask_idx = int(mask_seconds / dt)
        
        if mask_idx < len(Prediction):
            val_pred = Prediction[mask_idx:]
            val_exp = Output_exp[mask_idx:]
            if np.std(val_pred) > 1e-9:
                corr, _ = spearmanr(val_exp, val_pred)
            else:
                corr = 0.0
        print(f"Spearman Correlation (masked): {corr:.6f}")
    else:
        print("Kineticモデル計算失敗")
        Prediction = np.zeros_like(Output_exp)
        Input_plot = Input

    # ================= プロット作成 =================
    save_dir = os.path.join(RESULTS_DIR, "validation")
    os.makedirs(save_dir, exist_ok=True)
    
    total_time = len(Input_plot) * dt
    time_axis = np.arange(len(Input_plot)) * dt
    major_ticks = np.arange(0, int(total_time) + 1, 5) # 5秒刻み

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=False) 
    
    def setup_axis(ax, title, ylabel):
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, total_time)
        ax.set_xticks(major_ticks) 
        ax.grid(which='major', alpha=0.5, linestyle='-') 
        ax.tick_params(axis='x', labelbottom=True) 
        ax.set_xlabel('Time (s)', fontsize=10) 

    axes[0].plot(time_axis, Input_plot, color='gray', linewidth=1)
    setup_axis(axes[0], '1. Input Stimulus (Normalized)', 'Contrast')

    axes[1].plot(time_axis, g_t, color='blue', linewidth=1)
    setup_axis(axes[1], '2. Linear Filter Output g(t) [Normalized]', 'Filtered Signal')

    axes[2].plot(time_axis, u_t, color='green', linewidth=1)
    setup_axis(axes[2], '3. Nonlinear Output u(t)', 'Rate (u)')

    axes[3].plot(time_axis, Output_exp, color='black', alpha=0.5, label='Experiment', linewidth=1.5)
    axes[3].plot(time_axis, Prediction, color='red', alpha=0.8, label=f'Model (Corr={corr:.3f})', linewidth=1.5)
    setup_axis(axes[3], '4. Final Response (A state)', 'Response')
    axes[3].legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "detailed_analysis_fixed.png")
    plt.savefig(plot_path)
    print(f"詳細グラフを保存しました: {plot_path}")

if __name__ == "__main__":
    main()