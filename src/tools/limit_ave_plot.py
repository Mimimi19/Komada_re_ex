# -*- coding: utf-8 -*-
"""
src/tools/limit_ave_plot.py

ROI平均（例: data/ret2p/roi_ave/response_ave_roi1.txt）と、
limit 実験（scripts/limit/roi_1/band_low_only/seed_*/state/A_state.txt）
の30本を同一グラフに重ね、平均との差も表示して保存する。

- 予測は「state/A_state.txt」を使用（validation/predict.txt が無い環境でも動く）
- 正規化: 平均0、max|x|=1
- 出力: base_dir/ave_plot/ に png と LaTeX .tex

使い方:
  uv run python src/tools/limit_ave_plot.py
  uv run python src/tools/limit_ave_plot.py --base scripts/limit/roi_1/band_low_only --roi 1
"""
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント対応
try:
    import japanize_matplotlib
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

def save_latex_snippet(out_tex: str, png_name: str, title: str):
    tex = f"""\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=0.98\\linewidth]{{{png_name}}}
  \\caption{{{title}}}
  \\label{{fig:{os.path.splitext(os.path.basename(png_name))[0]}}}
\\end{{figure}}
"""
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)

def collect_astates(base_dir: str):
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

def main():
    # ==========================================================
    # ▼▼▼ ここを編集すれば、対象を切り替えられます ▼▼▼
    # ==========================================================
    DEFAULT_BASE = "scripts/limit/roi_1/band_low_only"
    DEFAULT_ROI = 1
    RESPONSE_AVE_TEMPLATE = "data/ret2p/roi_ave/response_ave_roi{roi}.txt"
    DT = 0.015625  # 64Hz 前提（固定）
    OUT_SUBDIR = "ave_plot"
    # ==========================================================

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=DEFAULT_BASE)
    ap.add_argument("--roi", type=int, default=DEFAULT_ROI)
    args = ap.parse_args()

    base_dir = args.base
    roi = args.roi

    if not os.path.isdir(base_dir):
        raise SystemExit(f"[Error] base_dir not found: {base_dir}")

    resp_path = RESPONSE_AVE_TEMPLATE.format(roi=roi)
    if not os.path.exists(resp_path):
        raise SystemExit(f"[Error] ROI average response not found: {resp_path}")

    resp = load_1d_txt(resp_path)
    traces, _seed_names = collect_astates(base_dir)

    if len(traces) == 0:
        raise SystemExit("[Error] No A_state.txt found under seed_*/state/")

    min_len = min([len(resp)] + [len(t) for t in traces])
    resp = resp[:min_len]
    traces = [t[:min_len] for t in traces]

    resp_n = normalize_zero_mean_max1(resp)
    traces_n = [normalize_zero_mean_max1(t) for t in traces]
    mean_pred = normalize_zero_mean_max1(np.mean(np.vstack(traces_n), axis=0))

    t_axis = np.arange(min_len) * DT

    out_dir = os.path.join(base_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_subplot(111)

    for tr in traces_n:
        ax.plot(t_axis, tr, color="green", alpha=0.5, linewidth=1.0)

    ax.plot(t_axis, mean_pred,color="red", alpha=1.0, linewidth=2.0, label="Pred mean (A_state mean)")
    ax.plot(t_axis, resp_n, color="blue", alpha=1.0, linewidth=2.0, label=f"Response ave ROI{roi}")

    ax.set_title(f"ROI{roi}: 30 optimized traces vs ROI-average response ({os.path.basename(base_dir)})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized (mean=0, max|x|=1)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, "limit_ave_plot.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    tex_path = os.path.join(out_dir, "limit_ave_plot.tex")
    save_latex_snippet(
        out_tex=tex_path,
        png_name="limit_ave_plot.pdf",
        title=f"ROI{roi}: overlay of 30 optimized A-state traces and ROI-average response.",
    )

    np.savetxt(os.path.join(out_dir, "pred_mean_normalized.txt"), mean_pred, fmt="%.6f")
    np.savetxt(os.path.join(out_dir, "response_ave_normalized.txt"), resp_n, fmt="%.6f")

    print("=== limit_ave_plot ===")
    print(f"base_dir : {base_dir}")
    print(f"roi      : {roi}")
    print(f"out_dir  : {out_dir}")
    print(f"saved    : {pdf_path}")
    print(f"saved    : {tex_path}")

if __name__ == "__main__":
    main()
