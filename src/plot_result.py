import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.stats import spearmanr
import glob

# 日本語フォント対応
try:
    import japanize_matplotlib
except ImportError:
    pass

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
    """パラメータ読み込み関数 (単一ファイルと個別ファイルの両方に対応)"""
    params = {}
    
    # パラメータ名の順序 (J個のalphasの直後から)
    keys_order = ['delta', 'a', 'kappa', 'b1', 'b2', 'ka', 'kfi', 'kfr', 'ksi', 'ksr', 
                  'p_R', 'p_A', 'p_I1', 'p_I2']
    
    # 1. 個別ファイル (L*.txt) があるかチェック
    l_files = glob.glob(os.path.join(results_dir, "L*.txt"))
    
    if l_files:
        # 個別ファイルからロード
        l_files.sort(key=lambda x: int(os.path.basename(x).replace('L', '').replace('.txt', '')))
        alphas = [float(np.loadtxt(f)) for f in l_files]
        params['alphas'] = np.array(alphas)
        params['J'] = len(alphas)
        
        for k in keys_order:
            p_path = os.path.join(results_dir, f"{k}.txt")
            if os.path.exists(p_path):
                params[k] = float(np.loadtxt(p_path))
            else:
                params[k] = 0.0 # デフォルト
        print(f"Loaded from individual files. J={params['J']}")
        
    else:
        # 2. 個別ファイルがない場合、epochsフォルダの最新ファイルをロード
        epoch_files = glob.glob(os.path.join(results_dir, "epochs", "epoch_*_params.txt"))
        if not epoch_files:
            # 念のためルートディレクトリの epoch_*.txt も探す
            epoch_files = glob.glob(os.path.join(results_dir, "epoch_*_params.txt"))
            
        if epoch_files:
            latest_file = max(epoch_files, key=lambda x: int(os.path.basename(x).split('_')[1]))
            print(f"Loading from single file: {latest_file}")
            vals = np.loadtxt(latest_file)
            
            # Jの推定: 全パラメータ数 - スカラーパラメータ数(14)
            J = len(vals) - 14
            params['J'] = J
            params['alphas'] = vals[0:J]
            
            for i, k in enumerate(keys_order):
                params[k] = float(vals[J + i])
        else:
            print("エラー: パラメータファイルが見つかりません。")
            sys.exit(1)

    return params

def normalize_states(p_R, p_A, p_I1, p_I2):
    total = p_R + p_A + p_I1 + p_I2
    if total > 1e-9:
        return p_R/total, p_A/total, p_I1/total, p_I2/total
    return 1.0, 0.0, 0.0, 0.0

def run_prediction(params, Input, dt, tau=1.0):
    """モデルを実行し、中間出力(g_t, u_t)も含めて返す"""
    R0, A0, I10, I20 = normalize_states(params['p_R'], params['p_A'], params['p_I1'], params['p_I2'])
    t_len = len(Input)

    # 1. Linear Filter
    linear_filter_kernel, _ = L_LNK.main(params['alphas'], params['delta'], t_len, dt, tau)
    
    g_full = fftconvolve(Input, linear_filter_kernel, mode='full')
    filter_len = int(tau / dt)
    shift_idx = len(linear_filter_kernel) - filter_len
    if shift_idx < 0: shift_idx = 0
    
    g_t = g_full[shift_idx : shift_idx + t_len]
    if len(g_t) < t_len: # パディング
        g_t = np.pad(g_t, (0, t_len - len(g_t)), 'constant')
    else:
        g_t = g_t[:t_len]

    # --- 【重要】ここが修正ポイント ---
    # 非線形関数に入力する前に g(t) を標準偏差1.0に正規化する
    # これにより、学習時と同じスケールで非線形関数が動作する
    g_std = np.std(g_t)
    if g_std > 1e-9:
        g_t = g_t / g_std
    else:
        print("警告: g_tの分散が0です。")
    # -----------------------------
    
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
    DATA_CONFIG_PATH = "config/data/ret2p-1.yaml"
    RESULTS_DIR = "scripts/20251212_22" 
    TAU = 1.0 
    # ============================================

    if len(sys.argv) > 1:
        RESULTS_DIR = sys.argv[1]
    
    if not os.path.exists(RESULTS_DIR):
        print(f"エラー: ディレクトリが見つかりません: {RESULTS_DIR}")
        return

    Input, Output_exp, dt = load_data(DATA_CONFIG_PATH)
    params = load_params_from_dir(RESULTS_DIR)

    min_len = min(len(Input), len(Output_exp))
    Input = Input[:min_len]
    Output_exp = Output_exp[:min_len]

    print("Running model prediction...")
    g_t, u_t, Prediction, check = run_prediction(params, Input, dt, tau=TAU)
    
    if check != 1:
        print("Kineticモデル計算失敗。")
        Prediction = np.zeros_like(Output_exp)
        corr = 0.0
    
    min_len_eval = min(len(Output_exp), len(Prediction), len(g_t), len(u_t))
    Output_exp = Output_exp[:min_len_eval]
    Prediction = Prediction[:min_len_eval]
    g_t = g_t[:min_len_eval]
    u_t = u_t[:min_len_eval]
    Input_plot = Input[:min_len_eval]

    # --- データの保存 (predict.txt用) ---
    # Predictionを「平均0, 最大値1」に変換
    if check == 1 and len(Prediction) > 0:
        Prediction = Prediction - np.mean(Prediction)
        p_max = np.max(Prediction)
        if p_max > 1e-9:
            Prediction = Prediction / p_max

    # --- マスクデータの作成 (masked_pre.txt用) ---
    masked_pre = Prediction.copy()
    mask_steps = int(1.0 / dt) # 最初の1秒
    if mask_steps > len(masked_pre): mask_steps = len(masked_pre)

    # 1秒以降の平均値で埋める
    if len(masked_pre) > mask_steps:
        fill_val = np.mean(masked_pre[mask_steps:])
    else:
        fill_val = 0.0
    masked_pre[:mask_steps] = fill_val
    
    # 再度「平均0, 最大値1」
    masked_pre = masked_pre - np.mean(masked_pre)
    mp_max = np.max(masked_pre)
    if mp_max > 1e-9:
        masked_pre = masked_pre / mp_max

    # 相関計算
    if check == 1:
        corr, _ = spearmanr(Output_exp, Prediction)
        print(f"Spearman Correlation: {corr:.6f}")

    # 保存
    save_dir = os.path.join(RESULTS_DIR, "validation")
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(os.path.join(save_dir, "g.txt"), g_t, fmt='%.6f')
    np.savetxt(os.path.join(save_dir, "u.txt"), u_t, fmt='%.6f')
    np.savetxt(os.path.join(save_dir, "predict.txt"), Prediction, fmt='%.6f')
    np.savetxt(os.path.join(save_dir, "masked_pre.txt"), masked_pre, fmt='%.6f')
    print(f"保存完了: {save_dir}")

    # プロット
    total_time = len(Input_plot) * dt
    time_axis = np.arange(len(Input_plot)) * dt
    major_ticks = np.arange(0, int(total_time) + 1, 1)

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=False) 
    def setup_axis(ax, title, ylabel):
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, total_time)
        ax.set_xticks(major_ticks) 
        ax.grid(which='major', alpha=0.5) 
        ax.set_xlabel('Time (s)', fontsize=10) 

    axes[0].plot(time_axis, Input_plot, color='gray', linewidth=1)
    setup_axis(axes[0], '1. Input Stimulus (Normalized)', 'Contrast')

    axes[1].plot(time_axis, g_t, color='blue', linewidth=1)
    setup_axis(axes[1], '2. Linear Filter Output g(t) [Normalized]', 'Filtered Signal')

    axes[2].plot(time_axis, u_t, color='green', linewidth=1)
    setup_axis(axes[2], '3. Nonlinear Output u(t)', 'Rate (u)')

    axes[3].plot(time_axis, Output_exp, color='black', alpha=0.5, label='Experiment', linewidth=1.5)
    axes[3].plot(time_axis, masked_pre, color='red', alpha=0.8, label=f'Model (Corr={corr:.3f})', linewidth=1.5)
    setup_axis(axes[3], '4. Final Response (Masked Prediction)', 'Response')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "detailed_analysis_normalized.png"))
    print("グラフ保存完了")

if __name__ == "__main__":
    main()