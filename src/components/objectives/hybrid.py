# src/components/objectives/hybrid.py
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr, pearsonr


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    if s < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n <= 0:
        return 1e6
    d = a[:n] - b[:n]
    return float(np.sqrt(np.mean(d * d)))


def _bandpass_zero_phase(x, dt, low_hz=None, high_hz=None, order=4):
    """
    zero-phase bandpass (sosfiltfilt).
    low_hz: None or >0  (high-pass edge)
    high_hz: None or <Nyquist (low-pass edge)
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy()

    fs = 1.0 / float(dt)
    nyq = fs * 0.5

    lo = None if low_hz is None else float(low_hz)
    hi = None if high_hz is None else float(high_hz)

    # sanitize
    if lo is not None and lo <= 0:
        lo = None
    if hi is not None and hi >= nyq:
        hi = nyq * 0.999

    # choose filter type
    if lo is None and hi is None:
        return x.copy()
    elif lo is None:
        # low-pass
        Wn = hi / nyq
        sos = butter(order, Wn, btype="lowpass", output="sos")
    elif hi is None:
        # high-pass
        Wn = lo / nyq
        sos = butter(order, Wn, btype="highpass", output="sos")
    else:
        # band-pass
        if not (0 < lo < hi < nyq):
            # invalid band → return original
            return x.copy()
        Wn = [lo / nyq, hi / nyq]
        sos = butter(order, Wn, btype="bandpass", output="sos")

    # filtfilt needs enough samples; if too short, fallback to no filtering
    try:
        return sosfiltfilt(sos, x)
    except Exception:
        return x.copy()


def _bandpass(x, dt, low_hz=None, high_hz=None, order=4):
    """
    Backward compatible wrapper.
    Accepts both positional and keyword usage in case old calls exist.
    """
    return _bandpass_zero_phase(x, dt, low_hz=low_hz, high_hz=high_hz, order=order)


def calculate(
    output,
    model,
    dt,
    objective_mode="hybrid",
    # weights (keep defaults conservative)
    w_rmse=0.0,
    w_pearson=0.0,
    w_band=1.0,
    order=4,
):
    """
    Returns a scalar to MINIMIZE.

    Modes:
      - "hybrid": (optional) RMSE - Pearson (classic)
      - "band_low_only":  maximize Spearman in 0.5–4 Hz  -> return -rho_low
      - "band_main_only": maximize Spearman in 4–30 Hz   -> return -rho_main
      - "band_full":      maximize weighted Spearman(low+main) (+ optional RMSE)
    """
    o = _zscore(np.asarray(output, dtype=float))
    m = _zscore(np.asarray(model, dtype=float))

    n = min(len(o), len(m))
    if n <= 3:
        return 5.0
    o = o[:n]
    m = m[:n]

    # --- band objectives ---
    if objective_mode in ("band_low_only", "band_main_only", "band_full"):
        # NOTE: fs=64Hzなら Nyquist=32Hz。main上限30HzはOK。highは基本評価不可(30–32の超狭帯域)。
        o_low = _bandpass_zero_phase(o, dt, low_hz=0.5, high_hz=4.0, order=order)
        m_low = _bandpass_zero_phase(m, dt, low_hz=0.5, high_hz=4.0, order=order)
        o_main = _bandpass_zero_phase(o, dt, low_hz=4.0, high_hz=30.0, order=order)
        m_main = _bandpass_zero_phase(m, dt, low_hz=4.0, high_hz=30.0, order=order)

        rho_low, _ = spearmanr(o_low, m_low)
        rho_main, _ = spearmanr(o_main, m_main)

        # NaN safety
        if not np.isfinite(rho_low):
            rho_low = 0.0
        if not np.isfinite(rho_main):
            rho_main = 0.0

        if objective_mode == "band_low_only":
            score = -float(rho_low) * float(w_band)
        elif objective_mode == "band_main_only":
            score = -float(rho_main) * float(w_band)
        else:
            # band_full: emphasize main, keep low as secondary
            score = -(1.0 * rho_main + 0.5 * rho_low) * float(w_band)

        # optional add-ons (usually keep small)
        if w_rmse and w_rmse > 0:
            score += float(w_rmse) * _rmse(o, m)

        if w_pearson and w_pearson > 0:
            r, _ = pearsonr(o, m)
            if not np.isfinite(r):
                r = 0.0
            score += float(w_pearson) * (1.0 - float(r))

        return float(score)

    # --- classic hybrid (fallback) ---
    rmse = _rmse(o, m)
    r, _ = pearsonr(o, m)
    if not np.isfinite(r):
        r = 0.0

    # minimize: RMSE - Pearson
    # (Pearson最大化したいのでマイナス)
    score = float(rmse) - float(r)
    return float(score)