# data/tools/block_visualization.py
# -*- coding: utf-8 -*-
"""
刺激波形と ROI 応答波形を可視化するツール。

出力:
  1. stimulus と ROI response を上下に並べた従来図
  2. stimulus と response を1つのグラフ内に重ねた図
     - 左軸: 刺激強度
     - 右軸: 応答強度

実行例:
uv run python data/tools/block_visualization.py scripts/limit \
    --resp-root data/ret2p/roi_ave \
    --stim data/ret2p/chirp_stim_64Hz_bilinear.txt \
    --colour true

色付けなし:
uv run python data/tools/block_visualization.py scripts/limit --colour false
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


# ==========================================================
# Block settings
# ==========================================================

BLOCKS = {
    "Block1":  (0.0, 10.0,  "lightskyblue"),
    "Block2":  (10.0, 20.0, "lightcoral"),
    "Block3":  (20.0, 30.0, "lightgreen"),
}


# ==========================================================
# ROI labels
# ==========================================================

ROI_LABELS = {
    1:  "CBC1 (OFF)",
    2:  "CBC2 (OFF)",
    3:  "CBC3a (OFF)",
    4:  "CBC3b (OFF)",
    5:  "CBC4 (OFF)",
    6:  "CBC5t (ON)",
    7:  "CBC5o (ON)",
    8:  "CBC5i (ON)",
    9:  "CBCX (ON)",
    10: "CBC6 (ON)",
    11: "CBC7 (ON)",
    12: "CBC8 (ON)",
    13: "CBC9 (ON)",
    14: "RBC (ON)",
}


def roi_label(roi_id: int) -> str:
    """
    ROI番号からPDF表示用ラベルを返す。
    未定義ROIの場合は ROI <id> として表示する。
    """
    return ROI_LABELS.get(roi_id, f"ROI {roi_id}")


# ==========================================================
# Utilities
# ==========================================================

def str2bool(v):
    """
    argparse 用 bool 変換。
    true/false, yes/no, 1/0 などを受け付ける。
    """
    if isinstance(v, bool):
        return v

    v = v.lower()

    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected. Use true or false.")


def _load_txt(path):
    return np.genfromtxt(path)


def draw_blocks(ax, colour=True):
    """
    Block1, Block2, Block3 の背景色を描画する。
    colour=False のときは何もしない。
    """
    if not colour:
        return

    for name, (t0, t1, color) in BLOCKS.items():
        ax.axvspan(t0, t1, color=color, alpha=0.3)


# ==========================================================
# Plot 1: separate panels
# ==========================================================

def plot_blocks_separate(stim, dt, roi_data, out_path, colour=True):
    """
    従来図。
    stimulus と ROI response を上下に分けて表示する。

    roi_data: dict { roi_id : response array }
    """
    N = len(roi_data) + 1  # +1 for stimulus
    t = np.arange(len(stim)) * dt

    fig, axes = plt.subplots(N, 1, figsize=(10, 2.5 * N), sharex=True)

    if N == 1:
        axes = [axes]

    # --- Stimulus ---
    ax = axes[0]
    draw_blocks(ax, colour=colour)
    ax.plot(t, stim, color="black", linewidth=1.0)
    ax.set_ylabel("照射刺激強度", fontsize=14)

    if colour:
        ax.set_title(
            "刺激波形とサブタイプごとの平均応答"
            "（Block1: 水色, Block2: 赤, Block3: 緑）",
            fontsize=18,
        )
    else:
        ax.set_title("刺激波形とサブタイプごとの平均応答", fontsize=18)

    ax.grid(True, alpha=0.3)

    # --- ROI responses ---
    for i, (roi_id, resp) in enumerate(roi_data.items(), start=1):
        label = roi_label(roi_id)

        ax = axes[i]
        draw_blocks(ax, colour=colour)
        ax.plot(
            t[:len(resp)],
            resp,
            linewidth=1.0,
            label=label,
        )

        ax.set_ylabel(f"{label}\nΔF/F [GluT]", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1, 1), loc="upper right", fontsize=14, framealpha=0.5)

    axes[-1].set_xlabel("Time (s)", fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_path}")


# ==========================================================
# Plot 2: stimulus-response overlay
# ==========================================================

def plot_stim_response_overlay(stim, dt, roi_data, out_path, colour=True):
    """
    追加図。
    stimulus と response を1つのグラフ内に重ねて表示する。

    左軸: stimulus
    右軸: response

    response は alpha=0.5 で描画する。
    """
    t = np.arange(len(stim)) * dt

    fig, ax_stim = plt.subplots(figsize=(11, 5))

    # background blocks
    draw_blocks(ax_stim, colour=colour)

    # --- left y-axis: stimulus ---
    stim_line, = ax_stim.plot(
        t,
        stim,
        color="black",
        linewidth=1.3,
        alpha=0.5,
        label="Stimulus",
    )

    ax_stim.set_xlabel("Time (s)", fontsize=14)
    # ax_stim.set_ylabel("照射刺激強度")
    ax_stim.set_ylabel("Light stimulus intensity", fontsize=14)
    ax_stim.grid(True, alpha=0.3)

    # --- right y-axis: responses ---
    ax_resp = ax_stim.twinx()

    response_lines = []

    for roi_id, resp in roi_data.items():
        label = roi_label(roi_id)

        line, = ax_resp.plot(
            t[:len(resp)],
            resp,
            linewidth=1.2,
            alpha=0.8,
            label=label,
        )
        response_lines.append(line)

    # ax_resp.set_ylabel("蛍光強度 ΔF/F [GluT]")
    ax_resp.set_ylabel("Fluorescence response ΔF/F [GluT]", fontsize=14)

    # legend
    lines = [stim_line] + response_lines
    labels = [line.get_label() for line in lines]

    ax_stim.legend(
        lines,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.79, 1.0),
        fontsize=14,
        framealpha=0.5,
    )

    if colour:
        ax_stim.set_title(
            "Stimulus and subtype responses overlaid with block highlights",
            fontsize=18,
        )
    else:
        ax_stim.set_title("Stimulus and subtype responses overlaid", fontsize=18)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_path}")


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "root_dir",
        help="例: scripts/limit",
    )
    parser.add_argument(
        "--resp-root",
        default="data/ret2p/roi_ave",
        help="ROI平均応答ファイルのディレクトリ",
    )
    parser.add_argument(
        "--stim",
        default="data/ret2p/chirp_stim_64Hz_bilinear.txt",
        help="刺激波形ファイル",
    )
    parser.add_argument(
        "--colour",
        type=str2bool,
        default=True,
        help="Block背景色を付けるか。true/false。default: true",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.015625,
        help="sampling interval [s]. default: 0.015625",
    )
    parser.add_argument(
        "--roi-list",
        type=str,
        default="1,14",
        help="表示するROI番号。例: 1,14 または 1,2,3,14",
    )

    args = parser.parse_args()

    # --- load stimulus ---
    stim = _load_txt(args.stim)
    dt = args.dt

    # --- ROI list ---
    roi_list = [int(x.strip()) for x in args.roi_list.split(",") if x.strip()]

    roi_data = {}

    for roi in roi_list:
        p = os.path.join(args.resp_root, f"response_ave_roi{roi}.txt")

        if not os.path.exists(p):
            print(f"[WARN] missing {p}")
            continue

        roi_data[roi] = _load_txt(p)

    if not roi_data:
        print("[ERROR] No ROI response data was loaded.")
        return

    # --- output directory ---
    out_dir = os.path.join(args.root_dir, "block_corr")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 従来図: 上下分割
    out_pdf_separate = os.path.join(
        out_dir,
        "stim_subtype_block_visualization_separate.pdf",
    )

    plot_blocks_separate(
        stim=stim,
        dt=dt,
        roi_data=roi_data,
        out_path=out_pdf_separate,
        colour=args.colour,
    )

    # 2. 追加図: 1グラフ内に重ねる
    out_pdf_overlay = os.path.join(
        out_dir,
        "stim_subtype_block_visualization_overlay_dual_axis.pdf",
    )

    plot_stim_response_overlay(
        stim=stim,
        dt=dt,
        roi_data=roi_data,
        out_path=out_pdf_overlay,
        colour=args.colour,
    )


if __name__ == "__main__":
    main()