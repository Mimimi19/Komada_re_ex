# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.signal import butter, filtfilt

def _bandpass(x: np.ndarray, dt: float, lo_hz: float, hi_hz: float, order: int = 4) -> np.ndarray:
    """
    Butterworth bandpass (zero-phase).
    dt: sampling interval [s]
    lo_hz, hi_hz: band edges [Hz]
    """
    if dt is None or dt <= 0:
        return x

    fs = 1.0 / dt
    nyq = 0.5 * fs

    lo = max(float(lo_hz), 1e-6)
    hi = min(float(hi_hz), nyq * 0.999)

    if hi <= lo:
        return x

    b, a = butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return filtfilt(b, a, x)

def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = spearmanr(a, b)
    if np.isnan(rho):
        return 0.0
    return float(rho)

def calculate(
    output_eval: np.ndarray,
    model_eval: np.ndarray,
    *,
    dt: float | None = None,
    objective_type: str = "hybrid",
    # hybrid weights
    w_rmse: float = 1.0,
    w_pearson: float = 0.5,
    # band objective settings
    band_low: tuple[float, float] = (0.5, 4.0),
    band_main: tuple[float, float] = (4.0, 30.0),
    filter_order: int = 4,
    # extra band term for "hybrid" mode
    w_band: float = 0.0,
    # legacy args (BaccusModel から渡される可能性があるが、ここでは未使用)
    use_diff_hp: bool = False,
):
    """
    返り値は「最小化すべきスコア」。

    - objective_type="hybrid":
        score = w_rmse*RMSE - w_pearson*Pearson - w_band*Spearman(main band)
      (RMSEは小さいほど良い / 相関は大きいほど良い → マイナスで入れる)

    - objective_type="band_low_only":
        score = - Spearman(0.5–4 Hz)

    - objective_type="band_main_only":
        score = - Spearman(4–30 Hz)

    - objective_type="band_full":
        score = - (0.5*Spearman(low) + 1.0*Spearman(main))
      ※重みは論文用の合理的デフォルト。必要ならここを調整。
    """
    output_eval = np.asarray(output_eval, dtype=np.float64)
    model_eval = np.asarray(model_eval, dtype=np.float64)

    # 平坦な波形は強ペナルティ（相関が不定になりやすい）
    if np.std(model_eval) <= 1e-9 or np.std(output_eval) <= 1e-9:
        return 10.0

    # --- band-only objectives ---
    if objective_type in ("band_low_only", "band_main_only", "band_full"):
        if dt is None:
            # dtが無いならbandpassできないので、全体Spearmanで代替
            rho = _safe_spearman(output_eval, model_eval)
            return -rho

        o_low = _bandpass(output_eval, dt, band_low[0], band_low[1], order=filter_order)
        m_low = _bandpass(model_eval, dt, band_low[0], band_low[1], order=filter_order)

        o_main = _bandpass(output_eval, dt, band_main[0], band_main[1], order=filter_order)
        m_main = _bandpass(model_eval, dt, band_main[0], band_main[1], order=filter_order)

        rho_low = _safe_spearman(o_low, m_low)
        rho_main = _safe_spearman(o_main, m_main)

        if objective_type == "band_low_only":
            return -rho_low
        if objective_type == "band_main_only":
            return -rho_main

        # band_full
        return -(0.5 * rho_low + 1.0 * rho_main)

    # --- default: hybrid (RMSE + Pearson + optional band Spearman) ---
    # 1. RMSE
    diff = output_eval - model_eval
    rmse = float(np.sqrt(np.mean(diff * diff)))

    # 2. Pearson
    r_val, _ = pearsonr(output_eval, model_eval)
    if np.isnan(r_val):
        r_val = 0.0

    # 3. Optional band Spearman（main band）
    rho_main = 0.0
    if (w_band is not None) and (w_band != 0.0) and (dt is not None):
        o_main = _bandpass(output_eval, dt, band_main[0], band_main[1], order=filter_order)
        m_main = _bandpass(model_eval, dt, band_main[0], band_main[1], order=filter_order)
        rho_main = _safe_spearman(o_main, m_main)

    # 相関を最大化 → スコアではマイナスで入れる
    score = (w_rmse * rmse) - (w_pearson * float(r_val)) - (float(w_band) * float(rho_main))
    return float(score)
