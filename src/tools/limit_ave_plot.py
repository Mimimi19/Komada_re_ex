# -*- coding: utf-8 -*-
"""
src/tools/limit_ave_plot.py

ROI平均（例: data/ret2p/roi_ave/response_ave_roi1.txt）と、
limit 実験（scripts/limit/roi_1/band_full/seed_*/state/A_state.txt）
の複数本を同一グラフに重ね、平均との差も表示して保存する。

- 予測は「state/A_state.txt」を使用
- 正規化: 平均0、max|x|=1
- 出力: scripts/limit/roi_{roi}/{objective}/ave_plot/ に pdf と LaTeX .tex を保存

使い方例:
  # ROI 1, band_full
  uv run python src/tools/limit_ave_plot.py --roi 1 --objective band_full

  # ROI 13, band_full
  uv run python src/tools/limit_ave_plot.py --roi 13 --objective band_full

  # ルートディレクトリを変えたい場合（必要なら）
  uv run python src/tools/limit_ave_plot.py --roi 1 --objective band_full --base-root scripts/limit_full
"""
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント対応
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


def normalize_zero_mean_max1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    m = np.max(np.abs(x))
    if m > 1e-12:
        x = x / m
    return x


def load_1d_txt(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=float)


def save_latex_snippet(out_tex: str, pdf_name: str, title: str):
    tex = f"""\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=0.98\\linewidth]{{{pdf_name}}}
  \\caption{{{title}}}
  \\label{{fig:{os.path.splitext(os.path.basename(pdf_name))[0]}}}
\\end{{figure}}
"""
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)


def collect_astates(base_dir: str):
    """
    base_dir/seed_*/state/A_state.txt をすべて読む
    """
    traces = []
    seed_names = []
    for seed_dir in sorted(glob.glob(os.path.join(base_dir, "seed_*"))):
        apath = os.path.join(seed_dir, "state", "A_state.txt")
        if not os.path.exists(apath):
            continue
        a = load_1d_txt(apath)
        traces.append(a)
        seed_names.append(os.path.basename(seed_dir))
    return traces, seed_names


def main(roi: int):
    # ==========================================================
    # ▼▼▼ ここの定数は基本いじらない想定 ▼▼▼
    # ==========================================================
    DEFAULT_BASE_ROOT = "scripts/limit"  # この直下に roi_1, roi_2, ... がある前提
    DEFAULT_ROI = roi if roi else 1
    DEFAULT_OBJECTIVE = "band_full"      # band_low_only / band_main_only / band_full など
    RESPONSE_AVE_TEMPLATE = "data/ret2p/roi_ave/response_ave_roi{roi}.txt"
    DT = 0.015625  # 64Hz 前提（固定）
    OUT_SUBDIR = "ave_plot"
    # ==========================================================
    # ROI → ラベル（x軸に表示する文字列）
    ROI_LABELS = {
        1:  "ROI 1  CBC1 (OFF)",
        2:  "ROI 2  CBC2 (OFF)",
        3:  "ROI 3  CBC3a (OFF)",
        4:  "ROI 4  CBC3b (OFF)",
        5:  "ROI 5  CBC4 (OFF)",
        6:  "ROI 6  CBC5t (ON)",
        7:  "ROI 7  CBC5o (ON)",
        8:  "ROI 8  CBC5i (ON)",
        9:  "ROI 9  CBCX (ON)",
        10: "ROI10 CBC6 (ON)",
        11: "ROI11 CBC7 (ON)",
        12: "ROI12 CBC8 (ON)",
        13: "ROI13 CBC9 (ON)",
        14: "ROI14 RBC (ON)",
    }
    # ==========================================================


    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", type=int, default=DEFAULT_ROI,
                    help="ROI番号（例: 1, 13, 14 ...）")
    ap.add_argument(
        "--objective",
        type=str,
        default=DEFAULT_OBJECTIVE,
        help="目的関数モードに対応したサブディレクトリ名 "
             "(例: band_low_only, band_main_only, band_full)",
    )
    ap.add_argument(
        "--base-root",
        type=str,
        default=DEFAULT_BASE_ROOT,
        help="roi_*/<objective> がぶら下がるルートディレクトリ (デフォルト: scripts/limit)",
    )
    args = ap.parse_args()

    roi = args.roi
    objective = args.objective
    base_root = args.base_root

    # base_dir = <base_root>/roi_{roi}/<objective>
    base_dir = os.path.join(base_root, f"roi_{roi}", objective)

    if not os.path.isdir(base_dir):
        raise SystemExit(f"[Error] base_dir not found: {base_dir}")

    # ROI 平均応答
    resp_path = RESPONSE_AVE_TEMPLATE.format(roi=roi)
    if not os.path.exists(resp_path):
        raise SystemExit(f"[Error] ROI average response not found: {resp_path}")

    resp = load_1d_txt(resp_path)

    # seed_* から A_state を集める
    traces, seed_names = collect_astates(base_dir)

    if len(traces) == 0:
        raise SystemExit("[Error] No A_state.txt found under seed_*/state/")

    # 長さを揃える
    min_len = min([len(resp)] + [len(t) for t in traces])
    resp = resp[:min_len]
    traces = [t[:min_len] for t in traces]

    # 正規化
    resp_n = normalize_zero_mean_max1(resp)
    traces_n = [normalize_zero_mean_max1(t) for t in traces]
    mean_pred = normalize_zero_mean_max1(np.mean(np.vstack(traces_n), axis=0))

    # 時間軸
    t_axis = np.arange(min_len) * DT

    # 出力ディレクトリ
    out_dir = os.path.join(base_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    # プロット
    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_subplot(111)

    # すべての seed を薄い緑で
    for tr in traces_n:
        ax.plot(t_axis, tr, color="green", alpha=0.5, linewidth=1.0)

    # 予測平均（赤）
    ax.plot(
        t_axis,
        mean_pred,
        color="red",
        alpha=1.0,
        linewidth=2.0,
        label="予測応答平均",
    )
    # ROI 平均応答（青）
    ax.plot(
        t_axis,
        resp_n,
        color="blue",
        alpha=1.0,
        linewidth=2.0,
        label=f"ROI{roi}の平均応答",
    )

    ax.set_title(
        f"{ROI_LABELS[roi]}における30試行の予測応答とROI平均応答の重ね合わせ "
    )
    ax.set_xlabel("時間 (s)")
    ax.set_ylabel("正規化後の振幅 (mean=0, max|x|=1)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, "limit_ave_plot.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # LaTeX figure snippet
    tex_path = os.path.join(out_dir, "limit_ave_plot.tex")
    save_latex_snippet(
        out_tex=tex_path,
        pdf_name="limit_ave_plot.pdf",
        title=f"ROI{roi}: overlay of optimized A-state traces and ROI-average response "
              f"({objective}).",
    )

    # 数値も保存（念のため）
    np.savetxt(os.path.join(out_dir, "pred_mean_normalized.txt"), mean_pred, fmt="%.6f")
    np.savetxt(os.path.join(out_dir, "response_ave_normalized.txt"), resp_n, fmt="%.6f")

    print("=== limit_ave_plot ===")
    print(f"base_root: {base_root}")
    print(f"base_dir : {base_dir}")
    print(f"roi      : {roi}")
    print(f"objective: {objective}")
    print(f"out_dir  : {out_dir}")
    print(f"saved    : {pdf_path}")
    print(f"saved    : {tex_path}")


if __name__ == "__main__":
    for roi in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        main(roi=roi)
