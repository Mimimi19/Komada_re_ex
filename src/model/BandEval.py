# src/model/BandEval.py
"""BandEval: band-limited evaluation helper

目的:
- response と predict（必要なら input も）を読み込み
- 指定帯域のフィルタを適用して Spearman / Pearson / RMSE を計算して表示
- データ長は固定でない前提で、BaccusModel と同様に min_len に揃えて評価

実行:
    uv run src/model/BandEval.py
"""

import os
import yaml
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import spearmanr, pearsonr


# ----------------------------
# I/O
# ----------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_series(path: str) -> np.ndarray:
    arr = np.genfromtxt(path)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float64)


def align_min_len(*arrays):
    min_len = min(len(a) for a in arrays if a is not None)
    out = []
    for a in arrays:
        out.append(None if a is None else a[:min_len])
    return out if len(out) > 1 else out[0]


# ----------------------------
# preprocessing
# ----------------------------
def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    s = np.std(x)
    if s < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


def _butter(dt: float, cutoff_hz: float, btype: str, order: int = 4):
    fs = 1.0 / dt
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    wn = min(max(wn, 1e-6), 0.999999)  # 安全クリップ
    return butter(order, wn, btype=btype)


def lowpass_filter(x: np.ndarray, dt: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if cutoff_hz is None:
        return x
    b, a = _butter(dt, cutoff_hz, btype="low", order=order)
    # filtfilt で短い系列が落ちるのを避ける
    if len(x) < max(32, 3 * (order + 1)):
        return x
    return filtfilt(b, a, x)


def highpass_filter(x: np.ndarray, dt: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if cutoff_hz is None:
        return x
    b, a = _butter(dt, cutoff_hz, btype="high", order=order)
    if len(x) < max(32, 3 * (order + 1)):
        return x
    return filtfilt(b, a, x)


# ----------------------------
# metrics
# ----------------------------
def evaluate(
    response: np.ndarray,
    predict: np.ndarray,
    dt: float,
    hp_hz: float | None,
    lp_hz: float | None,
    order: int = 4,
    mask_sec: float = 1.0,
    normalize: str = "zscore",
) -> dict:
    """指定の HP/LP を適用して、全体の指標を返す"""
    response, predict = align_min_len(response, predict)
    n_raw = len(response)

    y = response.copy()
    p = predict.copy()

    if hp_hz is not None and hp_hz > 0:
        y = highpass_filter(y, dt, hp_hz, order=order)
        p = highpass_filter(p, dt, hp_hz, order=order)
    if lp_hz is not None and lp_hz > 0:
        y = lowpass_filter(y, dt, lp_hz, order=order)
        p = lowpass_filter(p, dt, lp_hz, order=order)

    if normalize == "zscore":
        y = zscore(y)
        p = zscore(p)

    n_mask = int(mask_sec / dt) if (mask_sec is not None and mask_sec > 0) else 0
    if n_mask < len(y):
        y_eval = y[n_mask:]
        p_eval = p[n_mask:]
    else:
        y_eval = y
        p_eval = p

    n_eval = len(y_eval)

    if np.std(y_eval) < 1e-12 or np.std(p_eval) < 1e-12 or n_eval < 3:
        return {
            "n_raw": n_raw,
            "n_eval": n_eval,
            "spearman": float("nan"),
            "pearson": float("nan"),
            "rmse": float("nan"),
        }

    rho, _ = spearmanr(y_eval, p_eval)
    r, _ = pearsonr(y_eval, p_eval)
    rmse = float(np.sqrt(np.mean((y_eval - p_eval) ** 2)))

    return {
        "n_raw": n_raw,
        "n_eval": n_eval,
        "spearman": float(rho),
        "pearson": float(r),
        "rmse": rmse,
    }


def evaluate_bands(
    response: np.ndarray,
    predict: np.ndarray,
    dt: float,
    bands: dict,
    order: int = 4,
    mask_sec: float = 1.0,
    normalize: str = "zscore",
) -> dict:
    """複数帯域（low/main/high 等）の Spearman/Pearson を返す"""
    response, predict = align_min_len(response, predict)

    fs = 1.0 / dt
    nyq = 0.5 * fs

    out = {}
    for name, (lo, hi) in bands.items():
        lo_eff = None if lo is None else float(lo)
        hi_eff = None if hi is None else float(hi)

        # hi=None は Nyquist直下まで
        if hi_eff is None:
            hi_eff = nyq - 1e-6

        y = response.copy()
        p = predict.copy()

        # 「バンドパス」は HP→LP の直列で実現（lo/hiがNoneでもOK）
        if lo_eff is not None and lo_eff > 0:
            y = highpass_filter(y, dt, lo_eff, order=order)
            p = highpass_filter(p, dt, lo_eff, order=order)
        if hi_eff is not None and hi_eff > 0:
            y = lowpass_filter(y, dt, hi_eff, order=order)
            p = lowpass_filter(p, dt, hi_eff, order=order)

        if normalize == "zscore":
            y = zscore(y)
            p = zscore(p)

        n_mask = int(mask_sec / dt) if (mask_sec is not None and mask_sec > 0) else 0
        if n_mask < len(y):
            y_eval = y[n_mask:]
            p_eval = p[n_mask:]
        else:
            y_eval = y
            p_eval = p

        if np.std(y_eval) < 1e-12 or np.std(p_eval) < 1e-12 or len(y_eval) < 3:
            out[name] = {"spearman": float("nan"), "pearson": float("nan")}
            continue

        rho, _ = spearmanr(y_eval, p_eval)
        r, _ = pearsonr(y_eval, p_eval)
        out[name] = {"spearman": float(rho), "pearson": float(r)}
    return out


def main():
    # ==========================================================
    # ▼▼▼ ここを編集すれば、帯域と対象データを切り替えられます ▼▼▼
    # ==========================================================

    # A) data config（BaccusModel と同形式の yaml）
    USE_DATA_CONFIG = True
    DATA_CONFIG_PATH = "config/data/ret2p-1.yaml"  # input_file, output_file, dt を含む想定

    # B) 直指定（USE_DATA_CONFIG=False の場合）
    # INPUT_FILE = "data/ret2p/chirp_stim_64Hz_bilinear.txt"
    # RESPONSE_FILE = "data/ret2p/response_data_64Hz.txt"
    # DT = 0.015625

    # 予測（predict.txt）
    PREDICT_FILE = "scripts/ret2p/20251219_2302/validation/predict.txt"

    # 帯域（例：0.5Hz未満を切る & 30Hz超を切る）
    HP_HZ = 0.5
    LP_HZ = 30.0
    FILTER_ORDER = 4

    # 評価のマスク（秒）
    MASK_SEC = 1.0

    # 正規化
    NORMALIZE = "zscore"  # "zscore" or "none"

    # ==========================================================

    if USE_DATA_CONFIG:
        cfg = load_yaml(DATA_CONFIG_PATH)
        input_path = cfg["input_file"]
        response_path = cfg["output_file"]
        dt = float(cfg["dt"])
    else:
        input_path = INPUT_FILE
        response_path = RESPONSE_FILE
        dt = float(DT)

    fs = 1.0 / dt
    nyq = 0.5 * fs

    print("=== BandEval ===")
    print(f"Input    : {input_path}")
    print(f"Response : {response_path}")
    print(f"Predict  : {PREDICT_FILE}")
    print(f"dt       : {dt}  (fs={fs:.3f} Hz, Nyquist={nyq:.3f} Hz)")
    print(f"Filter   : HP={HP_HZ} Hz, LP={LP_HZ} Hz, order={FILTER_ORDER}")
    print(f"Mask     : {MASK_SEC} sec")
    print(f"Norm     : {NORMALIZE}")
    print("--------------")

    # 念のため input もロード（長さ確認用）
    _ = load_series(input_path)
    response = load_series(response_path)
    predict = load_series(PREDICT_FILE)

    stats = evaluate(
        response, predict, dt,
        hp_hz=HP_HZ, lp_hz=LP_HZ, order=FILTER_ORDER,
        mask_sec=MASK_SEC, normalize=NORMALIZE
    )

    print(f"Aligned length (raw) : {stats['n_raw']}")
    print(f"Eval length (masked) : {stats['n_eval']}")
    print(f"Spearman rho         : {stats['spearman']:.6f}")
    print(f"Pearson r            : {stats['pearson']:.6f}")
    print(f"RMSE                 : {stats['rmse']:.6f}")
    print("==============")

    # 低域(0.5-4), 主(4-30), 高域(30-Nyq) で Spearman/Pearson を出す
    bands = {
        "low": (0.5, 4.0),
        "main": (4.0, 30.0),
        "high": (30.0, None),  # Nyquist直下まで
    }

    band_stats = evaluate_bands(
        response, predict, dt,
        bands=bands,
        order=FILTER_ORDER,
        mask_sec=MASK_SEC,
        normalize=NORMALIZE
    )

    # 表示順を固定
    for k in ["low", "main", "high"]:
        if k not in band_stats:
            continue
        sp = band_stats[k]["spearman"]
        pr = band_stats[k]["pearson"]
        if np.isfinite(sp) and np.isfinite(pr):
            print(f"{k}:  Spearman={sp:.2f}, Pearson={pr:.2f}")
        else:
            print(f"{k}:  Spearman=nan, Pearson=nan")


if __name__ == "__main__":
    main()
