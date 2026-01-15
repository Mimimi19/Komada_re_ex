# data/tools/fft_spectrum.py
# -*- coding: utf-8 -*-
"""
ROI 1〜14 の FFT を個別 PDF & 全重ね PDF で保存する。

色付けは光の波長（赤→紫）に準拠し、
1〜4（赤系）, 5〜13（緑〜青系）, 14（紫）で明確に差を付けた。

出力先:
    data/ret2p/roi_ave/fft_figs/
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 色設定（ROI 1→14）
# -----------------------------
roi_colors = {
    1:  "#ff0000",  # 赤
    2:  "#ff7f00",  # 橙
    3:  "#ffff00",  # 黄
    4:  "#ccff00",  # 黄緑
    5:  "#00cc99",  # 緑
    6:  "#0099cc",
    7:  "#0066ff",
    8:  "#0033cc",  # 青
    9:  "#0000ff",
    10: "#3300cc",
    11: "#6600cc",
    12: "#9900ff",
    13: "#7f00ff",
    14: "#000000",  # 黒
}


def fft_power(x: np.ndarray, dt: float, fmax: float | None = None):
    """FFT → power spectrum."""
    x = np.asarray(x, dtype=float).flatten()
    mask = np.isfinite(x)
    x = x[mask]
    n = len(x)
    if n == 0:
        raise ValueError("入力データが空です")

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(X) ** 2

    if fmax is not None:
        idx = freqs <= fmax
        freqs = freqs[idx]
        power = power[idx]

    return freqs, power


def main():
    base_dir = "data/ret2p/roi_ave"
    out_dir = os.path.join(base_dir, "fft_figs")
    os.makedirs(out_dir, exist_ok=True)

    dt = 0.015625
    fmax = 40.0
    n_roi = 14

    all_freqs = None
    all_powers = []

    # =========================================
    # 個別 ROI 1〜14 の FFT 図を保存
    # =========================================
    for roi in range(1, n_roi + 1):
        fname = f"response_ave_roi{roi}.txt"
        in_path = os.path.join(base_dir, fname)

        if not os.path.isfile(in_path):
            print(f"[WARN] file not found: {in_path}")
            continue

        x = np.genfromtxt(in_path, dtype=float)
        freqs, power = fft_power(x, dt, fmax=fmax)

        if all_freqs is None:
            all_freqs = freqs

        all_powers.append((roi, power))

        # ---- 個別スペクトル ----
        plt.figure(figsize=(8, 5))
        plt.semilogy(freqs, power, color=roi_colors[roi], linewidth=2)

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power (log scale)")
        plt.title(f"FFT Power Spectrum (ROI {roi})", fontsize=12)
        plt.grid(True, linestyle="--")

        out_pdf = os.path.join(out_dir, f"response_ave_roi{roi}_fft_log_spectrum.pdf")
        plt.tight_layout()
        plt.savefig(out_pdf)
        plt.close()

        print(f"[OK] ROI {roi} → {out_pdf}")

    # =========================================
    # 全 ROI 重ね図
    # =========================================
    plt.figure(figsize=(9, 6))
    for roi, power in all_powers:
        plt.semilogy(all_freqs, power, color=roi_colors[roi],
                     alpha=0.5, linewidth=1.5, label=f"ROI {roi}")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power (log scale)")
    plt.title("FFT Power Spectrum (All ROI)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(ncol=2, fontsize=8)

    out_pdf_all = os.path.join(out_dir, "roi1-14_fft_log_spectrum_all.pdf")
    plt.tight_layout()
    plt.savefig(out_pdf_all)
    plt.close()

    print("\n=== DONE ===")
    print(f"All ROI figure → {out_pdf_all}")


if __name__ == "__main__":
    main()
