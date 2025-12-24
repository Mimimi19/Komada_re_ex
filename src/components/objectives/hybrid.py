# src/components/objectives/hybrid.py
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.signal import butter, filtfilt


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    if s < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


def _bandpass_zero_phase(x: np.ndarray, dt: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """
    0位相(BW) bandpass: butter + filtfilt
    dt: sampling interval [sec]
    """
    x = np.asarray(x, dtype=float)
    fs = 1.0 / dt
    nyq = 0.5 * fs

    # 安全域に丸める（Nyquist超え・ゼロ帯域を回避）
    lo = max(1e-6, float(low_hz))
    hi = float(high_hz)
    hi = min(hi, nyq * 0.999)

    if hi <= lo:
        # 帯域が成立しないならそのまま返す（目的関数のペナルティ側で扱う想定）
        return x

    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    # filtfilt は短すぎる系列で落ちるのでガード
    if x.size < (3 * max(len(a), len(b))):
        return x
    return filtfilt(b, a, x)


def _bandpass(*args, **kwargs) -> np.ndarray:
    """
    互換ラッパ：
      _bandpass(x, dt, low_hz=..., high_hz=..., order=...)
    だけでなく
      _bandpass(x, dt, low=..., high=...)
    も受ける（古い呼び出しの吸収）
    """
    if len(args) < 2:
        raise TypeError("_bandpass(x, dt, ...) が必要です")
    x = args[0]
    dt = args[1]

    low_hz = kwargs.pop("low_hz", None)
    high_hz = kwargs.pop("high_hz", None)
    if low_hz is None:
        low_hz = kwargs.pop("low", None)
    if high_hz is None:
        high_hz = kwargs.pop("high", None)
    order = kwargs.pop("order", 4)

    if low_hz is None or high_hz is None:
        raise TypeError("_bandpass は low_hz/high_hz（または low/high）が必要です")

    return _bandpass_zero_phase(np.asarray(x, dtype=float), float(dt), float(low_hz), float(high_hz), int(order))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return 1e9
    d = a[:n] - b[:n]
    return float(np.sqrt(np.mean(d * d)))


def calculate(output: np.ndarray,
              model: np.ndarray,
              dt: float,
              objective_type: str = "hybrid",
              # hybrid 用
              w_rmse: float = 1.0,
              w_pearson: float = 1.0,
              # band Spearman 用
              band_low_hz: float = 0.5,
              band_main_low_hz: float = 4.0,
              band_main_high_hz: float = 30.0,
              band_high_hz: float = 30.0,
              band_order: int = 4,
              # 全帯域/補助
              use_diff_hp: bool = False,
              w_band: float = 2.0,
              ) -> float:
    """
    返り値は **最小化すべきスコア**（小さいほど良い）

    objective_type:
      - "band_low_only"  : 0.5–4Hz の Spearman を最大化（score = -rho）
      - "band_main_only" : 4–30Hz の Spearman を最大化（score = -rho）
      - "band_full"      : low + main を同時最大化（score = -(rho_low + rho_main)/2）
      - "hybrid"         : 既存（RMSE + Pearson + band penalty など）
    """
    o = np.asarray(output, dtype=float)
    m = np.asarray(model, dtype=float)
    n = min(o.size, m.size)
    if n < 8:
        return 5.0
    o = o[:n]
    m = m[:n]

    # ---- band objectives ----
    if objective_type in ("band_low_only", "band_main_only", "band_full"):
        # z-score してから bandpass（BandEval と同じ思想）
        o0 = _safe_zscore(o)
        m0 = _safe_zscore(m)

        # low: 0.5–4
        o_low = _bandpass(o0, dt, low_hz=band_low_hz, high_hz=band_main_low_hz, order=band_order)
        m_low = _bandpass(m0, dt, low_hz=band_low_hz, high_hz=band_main_low_hz, order=band_order)

        # main: 4–30
        o_main = _bandpass(o0, dt, low_hz=band_main_low_hz, high_hz=band_main_high_hz, order=band_order)
        m_main = _bandpass(m0, dt, low_hz=band_main_low_hz, high_hz=band_main_high_hz, order=band_order)

        # Spearman
        rho_low, _ = spearmanr(o_low, m_low)
        rho_main, _ = spearmanr(o_main, m_main)
        if not np.isfinite(rho_low):
            rho_low = 0.0
        if not np.isfinite(rho_main):
            rho_main = 0.0

        if objective_type == "band_low_only":
            return float(-rho_low)
        if objective_type == "band_main_only":
            return float(-rho_main)

        # band_full
        return float(-0.5 * (rho_low + rho_main))

    # ---- default hybrid (既存互換) ----
    # 注意：ここはあなたの既存 hybrid 実装に合わせて調整してOK。
    # 最低限「落ちない」形で入れてあります。
    o0 = _safe_zscore(o)
    m0 = _safe_zscore(m)

    rmse = _rmse(o0, m0)
    pr, _ = pearsonr(o0, m0)
    if not np.isfinite(pr):
        pr = 0.0

    # optional: 4–30 band Spearman を penalty として追加
    o_bp = _bandpass(o0, dt, low_hz=band_main_low_hz, high_hz=band_main_high_hz, order=band_order)
    m_bp = _bandpass(m0, dt, low_hz=band_main_low_hz, high_hz=band_main_high_hz, order=band_order)
    rho_band, _ = spearmanr(o_bp, m_bp)
    if not np.isfinite(rho_band):
        rho_band = 0.0

    # 最小化：rmse は小、相関は大が良いので符号反転
    score = (w_rmse * rmse) + (w_pearson * (1.0 - pr)) + (w_band * (1.0 - rho_band))
    return float(score)