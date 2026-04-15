# data/tools/block_visualization.py
# -*- coding: utf-8 -*-
"""
uv run data/tools/block_visualization.py \
    --root-dir scripts/limit \
    --resp-root data/ret2p/roi_ave \
    --stim data/ret2p/chirp_stim_64Hz_bilinear.txt
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib
except ImportError:
    pass


BLOCKS = {
    "Block1":  (0.0, 10.0,  'lightskyblue'),
    "Block2":  (10.0, 20.0, 'lightcoral'),
    "Block3":  (20.0, 30.0, 'lightgreen'),
    "Block23": (10.0, 30.0, 'lightgray'),  # background highlight only
}


def _load_txt(path):
    return np.genfromtxt(path)


def plot_blocks(stim, dt, roi_data, out_path):
    """
    roi_data: dict { roi_id : response array }
              → 今回は {1: resp1, 14: resp14 }
    """
    N = len(roi_data) + 1  # +1 for stimulus
    t = np.arange(len(stim)) * dt

    fig, axes = plt.subplots(N, 1, figsize=(10, 2.5*N), sharex=True)

    def draw_blocks(ax):
        for name, (t0, t1, color) in BLOCKS.items():
            ax.axvspan(t0, t1, color=color, alpha=0.3)

    # --- Stimulus ---
    ax = axes[0]
    draw_blocks(ax)
    ax.plot(t, stim, color='black', linewidth=1.0)
    ax.set_ylabel("照射刺激強度")
    ax.set_title("刺激波形と ROIごとの平均応答（Block1: 水色, Block2: 赤, Block3: 緑）")
    ax.grid(True, alpha=0.3)

    # --- ROI ---
    for i, (roi_id, resp) in enumerate(roi_data.items(), start=1):
        ax = axes[i]
        draw_blocks(ax)
        ax.plot(t[:len(resp)], resp, linewidth=1.0)
        ax.set_ylabel(f"ROI {roi_id}の蛍光強度(ΔF/F)[Gult]")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="例: scripts/limit")
    parser.add_argument("--resp-root", default="data/ret2p/roi_ave")
    parser.add_argument("--stim", default="data/ret2p/chirp_stim_64Hz_bilinear.txt")
    args = parser.parse_args()

    # load stim
    stim = _load_txt(args.stim)
    dt = 0.015625  # 64Hz

    roi_data = {}
    for roi in (1, 14):
        p = os.path.join(args.resp_root, f"response_ave_roi{roi}.txt")
        if not os.path.exists(p):
            print(f"[WARN] missing {p}")
            continue
        roi_data[roi] = _load_txt(p)

    out_dir = os.path.join(args.root_dir, "block_corr")
    os.makedirs(out_dir, exist_ok=True)

    out_pdf = os.path.join(out_dir, "stim_roi1_roi14_block_visualization.pdf")
    plot_blocks(stim, dt, roi_data, out_pdf)


if __name__ == "__main__":
    main()
