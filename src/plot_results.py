import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.stats import spearmanr
import glob
import re

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
    """パラメータ読み込み関数 (epochファイルを優先)"""
    params = {}
    
    # BaccusModel.py の変更に合わせて、初期値パラメータ(p_R等)を削除
    # J個のalphasの後に続くスカラーパラメータのリスト
    keys_order = [
        'delta', 'a', 'kappa', 'b1', 'b2', 
        'ka', 'kfi', 'kfr', 'ksi', 'ksr', 
        'w_gain', 'w_decay'
    ]
    num_scalar_params = len(keys_order) # 12個
    
    # 1. まず epoch_*.txt (まとまったファイル) を探す
    epoch_files = glob.glob(os.path.join(results_dir, "epochs", "epoch_*_params.txt"))
    if not epoch_files:
        epoch_files = glob.glob(os.path.join(results_dir, "epoch_*_params.txt"))
        
    if epoch_files:
        # 数字部分を抽出してソートし、最新のファイルを取得
        def extract_epoch_num(path):
            match = re.search(r'epoch_(\d+)_params', path)
            return int(match.group(1)) if match else 0
            
        latest_file = max(epoch_files, key=extract_epoch_num)
        print(f"Loading from single file (PRIORITY): {latest_file}")
        vals = np.loadtxt(latest_file)
        
        # Jの推定: 全パラメータ数 - スカラーパラメータ数
        if len(vals) < num_scalar_params:
             print("エラー: パラメータファイルが短すぎます。")
             sys.exit(1)
             
        J = len(vals) - num_scalar_params
        params['J'] = J
        params['alphas'] = vals[0:J]
        
        for i, k in enumerate(keys_order):
            val = float(vals[J + i])
            # kaのゼロ除算対策
            if k == 'ka' and abs(val) < 1e-9:
                val = 1.0
            params[k] = val

    # 2. epochファイルがない場合のみ、個別ファイル (L*.txt) を探す
    else:
        l_files = glob.glob(os.path.join(results_dir, "L*.txt"))
        
        if l_files:
            l_files.sort(key=lambda x: int(os.path.basename(x).replace('L', '').replace('.txt', '')))
            alphas = [float(np.loadtxt(f)) for f in l_files]
            params['alphas'] = np.array(alphas)
            params['J'] = len(alphas)
            
            for k in keys_order:
                p_path = os.path.join(results_dir, f"{k}.txt")
                if os.path.exists(p_path):
                    val = float(np.loadtxt(p_path))
                    if k == 'ka' and abs(val) < 1e-9:
                        print(f"警告: {k} が 0.0 に近いため 1.0 に補正します。")
                        val = 1.0
                    params[k] = val
                else:
                    if k == 'ka':
                        params[k] = 1.0
                    elif k == 'kappa':
                         params[k] = 1.0
                    else:
                        params[k] = 0.0 
            print(f"Loaded from individual files. J={params['J']}")
        else:
            print("エラー: パラメータファイルが見つかりません。")
            sys.exit(1)

    return params

def calculate_steady_state(params):
    """
    パラメータから平均入力に対する定常状態(R, A, I1, I2)を計算する。
    BaccusModel.py の _calculate_steady_state と同じロジック。
    """
    a = params['a']
    kappa = params['kappa']
    b1 = params['b1']
    b2 = params['b2']
    ka = params['ka']
    kfi = params['kfi']
    kfr = params['kfr']
    ksi = params['ksi']
    ksr = params['ksr']

    # 1. 定常入力 u_steady の計算 (入力=0 のときの非線形応答)
    # N_LNK.main は配列を返すので [0] を取得
    u_steady = N_LNK.main(np.array([0.0]), a, kappa, b1, b2, ka)[0]
    u_steady = max(0.0, u_steady)

    # 2. 定常状態の代数的計算
    if kfi > 1e-9:
        A_ratio = (ka * u_steady) / kfi
    else:
        A_ratio = 0.0
        
    if kfr > 1e-9:
        I1_ratio = (ka * u_steady) / kfr
    else:
        I1_ratio = 0.0

    if kfr > 1e-9 and ksr > 1e-9:
         # I2への分岐がある場合 (use_I2=True前提で計算)
         # I2 = (ksi * I1) / (ksr * u) -> I2 = (ksi * ka) / (kfr * ksr)
        I2_ratio = (ksi * ka) / (kfr * ksr)
    else:
        I2_ratio = 0.0

    # 3. 合計が1になるように正規化
    total = 1.0 + A_ratio + I1_ratio + I2_ratio
    
    return (1.0/total), (A_ratio/total), (I1_ratio/total), (I2_ratio/total)

def run_prediction(params, Input, dt, tau=1.0):
    """
    モデルを実行し、g_t, u_t, W (電流) を返す
    """
    # 初期状態をパラメータから計算
    R0, A0, I10, I20 = calculate_steady_state(params)
    
    # 1. Linear Filter
    filter_points = int(tau / dt) + 1
    linear_filter_kernel, _ = L_LNK.main(params['alphas'], params['delta'], filter_points, dt, tau)
    
    g_full = fftconvolve(Input, linear_filter_kernel, mode='full')
    shift_idx = int(tau / dt) 
    
    if len(g_full) > shift_idx + len(Input):
        g_t = g_full[shift_idx : shift_idx + len(Input)]
    else:
        g_t = g_full[:len(Input)]
    
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
    R, A, I1, I2, W, check = K_LNK.main(
        len(u_t), u_t, dt, 
        R0, A0, I10, I20,
        params['ka'], params['kfi'], params['kfr'], params['ksi'], params['ksr'],
        params['w_gain'], params['w_decay'],
        label="Validation"
    )
    
    return g_t, u_t, W, check

def main(data_config_path, results_dir, tau=1.0):
    if not os.path.exists(results_dir):
        print(f"エラー: ディレクトリが見つかりません: {results_dir}")
        return

    # データロード
    Input, Output_exp, dt = load_data(data_config_path)
    
    # パラメータロード
    print(f"Loading parameters from {results_dir}...")
    try:
        params = load_params_from_dir(results_dir)
    except Exception as e:
        print(f"パラメータ読み込みエラー: {e}")
        return

    min_len = min(len(Input), len(Output_exp))
    Input = Input[:min_len]
    Output_exp = Output_exp[:min_len]

    print("Running model prediction...")
    # W_raw (生の電流値) を受け取る
    g_t, u_t, W_raw, check = run_prediction(params, Input, dt, tau=tau)
    
    if check != 1:
        print("Kineticモデル計算失敗。")
        W_raw = np.zeros_like(Output_exp)
        corr = 0.0
    
    min_len_eval = min(len(Output_exp), len(W_raw), len(g_t), len(u_t))
    Output_exp = Output_exp[:min_len_eval]
    W_raw = W_raw[:min_len_eval]
    g_t = g_t[:min_len_eval]
    u_t = u_t[:min_len_eval]
    Input_plot = Input[:min_len_eval]

    # --- データの保存 ---
    
    # 1. predict.txt (相関計算用の正規化後データ)
    # DE_Simulationでは (-1 * w) と相関を見ていたので、ここでも反転させる
    Prediction = -1.0 * W_raw 
    
    # 平均0, 最大絶対値1に正規化 (比較用)
    Prediction = Prediction - np.mean(Prediction)
    p_max = np.max(np.abs(Prediction))
    if p_max > 1e-9:
        Prediction = Prediction / p_max

    # 2. masked_pre.txt (マスク済み)
    masked_pre = Prediction.copy()
    mask_steps = int(1.0 / dt) # 1秒マスク
    if mask_steps > len(masked_pre): mask_steps = len(masked_pre)

    if len(masked_pre) > mask_steps:
        fill_val = np.mean(masked_pre[mask_steps:])
    else:
        fill_val = 0.0
    masked_pre[:mask_steps] = fill_val
    
    # 相関計算 (マスク済みデータで計算)
    if check == 1:
        if len(Output_exp) > mask_steps:
             corr, _ = spearmanr(Output_exp[mask_steps:], masked_pre[mask_steps:])
        else:
             corr, _ = spearmanr(Output_exp, masked_pre)
        print(f"Spearman Correlation (Masked): {corr:.6f}")
    else:
        corr = 0.0

    # 保存
    save_dir = os.path.join(results_dir, "validation")
    os.makedirs(save_dir, exist_ok=True)
    
    np.savetxt(os.path.join(save_dir, "g.txt"), g_t, fmt='%.6f')
    np.savetxt(os.path.join(save_dir, "u.txt"), u_t, fmt='%.6f')
    
    # r.txt には生の出力 (電流 W_raw) を保存
    np.savetxt(os.path.join(save_dir, "r.txt"), W_raw, fmt='%.6f')
    
    # predict.txt には正規化後の予測応答
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

    # 4段目: 実測値 vs 予測値 (電流)
    axes[3].plot(time_axis, Output_exp, color='black', alpha=0.5, label='Experiment', linewidth=1.5)
    axes[3].plot(time_axis, masked_pre, color='red', alpha=0.8, label=f'Model Current (Corr={corr:.3f})', linewidth=1.5)
    setup_axis(axes[3], '4. Final Response (Synaptic Current W)', 'Response')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "detailed_analysis_current.png"))
    print("グラフ保存完了")

if __name__ == "__main__":
    # =================================================================
    # ▼▼▼ ここに使用するデータと結果フォルダの詳細を入力してください ▼▼▼
    # =================================================================
    
    # 1. データ設定ファイル (.yaml) のパス
    # TARGET_CONFIG = "config/data/Ucb1.yaml"
    TARGET_CONFIG = "config/data/Ucb2.yaml"
    # TARGET_CONFIG = "config/data/ret2p-1.yaml"
    
    # 2. 学習結果が保存されているディレクトリ
    # TARGET_DIR = "scripts/cb1/20251218_1012"
    TARGET_DIR = "scripts/ret2p/20251218_1906/1_segment"
    # TARGET_DIR = "scripts/ret2p/20251217_23"
    
    # 3. 時定数 (Tau)
    TARGET_TAU = 1.0

    # =================================================================
    
    # コマンドライン引数がある場合はそちらを優先
    if len(sys.argv) > 1:
        TARGET_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        TARGET_CONFIG = sys.argv[2]

    print(f"--- 実行設定 ---")
    print(f"Config : {TARGET_CONFIG}")
    print(f"Results: {TARGET_DIR}")
    print(f"Tau    : {TARGET_TAU}")
    print(f"---------------")

    # main関数に設定を渡して実行
    main(TARGET_CONFIG, TARGET_DIR, TARGET_TAU)