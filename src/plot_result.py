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
    """
    R0, A0, I10, I20 = normalize_states(params['p_R'], params['p_A'], params['p_I1'], params['p_I2'])
    t_len = len(Input)

    # 1. Linear Filter
    linear_filter_kernel, _ = L_LNK.main(params['alphas'], params['delta'], t_len, dt, tau)
    # 線形フィルタ出力 g(t)
    g_t = fftconvolve(Input, linear_filter_kernel, mode='same')
    
    # 2. Nonlinear Module
    # 非線形変換後の出力 u(t)
    u_t = N_LNK.main(
        g_t, 
        params['a'], 
        params['kappa'], 
        params['b1'], 
        params['b2'], 
        params['ka']
    )
    
    # 3. Kinetic Model
    # 最終応答 A(t)
    R, A, I1, I2, check = K_LNK.main(
        len(u_t), u_t, dt, 
        R0, A0, I10, I20,
        params['ka'], params['kfi'], params['kfr'], params['ksi'], params['ksr'],
        label="Validation"
    )
    
    # 中間出力も返す
    return g_t, u_t, A, check

def main():
    # ================= 設定エリア =================
    DATA_CONFIG_PATH = "config/data/ret2p-1.yaml"
     # 例: "scripts/results/Baccus_ret2p/20251211_14"
    # RESULTS_DIR = "scripts/results/Baccus_ret2p/20251211_07" 
    RESULTS_DIR = "scripts/20251211_07" 
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
    
    # エラーハンドリング
    corr = 0.0
    if check != 1:
        print("\n!!!!!!!! 警告 !!!!!!!!")
        print("Kineticモデルの計算が失敗しました (check != 1)。")
        print("Final Response (A) をゼロ埋めして表示します。")
        Prediction = np.zeros_like(Output_exp)
    else:
        # 長さ調整
        min_len_eval = min(len(Output_exp), len(Prediction))
        Output_exp = Output_exp[:min_len_eval]
        Prediction = Prediction[:min_len_eval]
        g_t = g_t[:min_len_eval]
        u_t = u_t[:min_len_eval]
        
        corr, _ = spearmanr(Output_exp, Prediction)
        print(f"Spearman Correlation: {corr:.6f}")

    # ================= プロット作成 (4段構成) =================
   # ================= プロット作成 (4段構成) =================
    save_dir = os.path.join(RESULTS_DIR, "validation")
    os.makedirs(save_dir, exist_ok=True)
    
    # 時間軸の作成
    total_time = len(Input) * dt
    time_axis = np.arange(len(Input)) * dt

    # 1秒刻みの目盛りを作成 (0, 1, 2, ..., total_time)
    major_ticks = np.arange(0, int(total_time) + 1, 1)

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=False) # sharex=Falseにして個別に制御
    
    # 共通のスタイル設定関数
    def setup_axis(ax, title, ylabel):
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, total_time)
        ax.set_xticks(major_ticks) # 1秒単位の目盛りを設定
        ax.grid(which='major', alpha=0.5, linestyle='-') # グリッド線
        ax.tick_params(axis='x', labelbottom=True) # すべてのグラフでX軸ラベルを表示
        ax.set_xlabel('Time (s)', fontsize=10) # すべてにラベルをつける

    # 1. Input Stimulus
    axes[0].plot(time_axis, Input, color='gray', linewidth=1)
    setup_axis(axes[0], '1. Input Stimulus (Normalized)', 'Contrast')

    # 2. Linear Filter Output (g_t)
    axes[1].plot(time_axis, g_t, color='blue', linewidth=1)
    setup_axis(axes[1], '2. Linear Filter Output g(t)', 'Filtered Signal')

    # 3. Nonlinear Output (u_t)
    axes[2].plot(time_axis, u_t, color='green', linewidth=1)
    setup_axis(axes[2], '3. Nonlinear Output u(t) (Rate Constant)', 'Rate (u)')

    # 4. Final Response (A state)
    axes[3].plot(time_axis, Output_exp, color='black', alpha=0.5, label='Experiment', linewidth=1.5)
    axes[3].plot(time_axis, Prediction, color='red', alpha=0.8, label=f'Model (Corr={corr:.3f})', linewidth=1.5)
    setup_axis(axes[3], '4. Final Response (Calcium/Fluorescence)', 'Response')
    axes[3].legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "detailed_analysis.png")
    plt.savefig(plot_path)
    print(f"詳細グラフを保存しました: {plot_path}")
    
    # plt.show() # サーバー環境ならコメントアウト推奨

if __name__ == "__main__":
    main()