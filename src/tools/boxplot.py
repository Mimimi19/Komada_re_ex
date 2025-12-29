# -*- coding: utf-8 -*-
"""
src/tools/boxplot.py

- scripts/limit/roi_1/band_low_only/seed_01/correlation.txt のようなファイルを集計
- objective（band_low_only 等）ごとに seed 分布を箱ひげ図で可視化
- PNG と LaTeX で貼れる figure .tex を出力

使い方:
  uv run python src/tools/boxplot.py
  uv run python src/tools/boxplot.py --base scripts/limit/roi_1/band_low_only
"""
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

def _read_float(path: str):
    try:
        s = open(path, "r", encoding="utf-8").read().strip()
        # "corr: 0.123" みたいな形式にも対応
        for tok in s.replace(",", " ").split():
            try:
                return float(tok)
            except Exception:
                pass
        return float(s)
    except Exception:
        return None

def collect_correlations(base_dir: str):
    """
    base_dir/
      seed_01/correlation.txt
      seed_02/correlation.txt
      ...
    を探索して読み取る
    """
    corr = []
    seeds = []
    for seed_dir in sorted(glob.glob(os.path.join(base_dir, "seed_*"))):
        cpath = os.path.join(seed_dir, "correlation.txt")
        if not os.path.exists(cpath):
            continue
        v = _read_float(cpath)
        if v is None or not np.isfinite(v):
            continue
        corr.append(float(v))
        seeds.append(os.path.basename(seed_dir))
    return np.array(corr, dtype=float), seeds

def save_latex_snippet(out_tex: str, png_name: str, title: str):
    """\includegraphics で貼れる最小 figure を生成"""
    tex = f"""\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=0.9\\linewidth]{{{png_name}}}
  \\caption{{{title}}}
  \\label{{fig:{os.path.splitext(os.path.basename(png_name))[0]}}}
\\end{{figure}}
"""
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)

def main():
    # ==========================================================
    # ▼▼▼ ここを編集すれば、対象ディレクトリを切り替えられます ▼▼▼
    # ==========================================================
    DEFAULT_BASE = "scripts/limit/roi_1/band_low_only"
    OUT_SUBDIR = "boxplot"
    # ==========================================================

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=DEFAULT_BASE, help="seed_*/correlation.txt が入ったディレクトリ")
    args = ap.parse_args()

    base_dir = args.base
    if not os.path.isdir(base_dir):
        raise SystemExit(f"[Error] base_dir not found: {base_dir}")

    corr, _seeds = collect_correlations(base_dir)
    if corr.size == 0:
        raise SystemExit("[Error] No correlation.txt found under seed_*")
    out_dir = os.path.join(base_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    best = float(np.max(corr))
    med  = float(np.median(corr))
    p95  = float(np.quantile(corr, 0.95))
    mean = float(np.mean(corr))

    # 保存（数値）
    stats_path = os.path.join(out_dir, "summary_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"n={corr.size}\n")
        f.write(f"best={best:.6f}\n")
        f.write(f"mean={mean:.6f}\n")
        f.write(f"median={med:.6f}\n")
        f.write(f"p95={p95:.6f}\n")

    # プロット
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.boxplot(corr, vert=True, showmeans=True)
    ax.set_title("Objective distribution across seeds")
    ax.set_ylabel("Objective value (as saved in correlation.txt)")
    ax.set_xticks([1])
    ax.set_xticklabels([os.path.basename(base_dir)])

    txt = f"n={corr.size}  best={best:.3f}  median={med:.3f}  p95={p95:.3f}"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "boxplot.png")
    plt.savefig(png_path, dpi=200)
    plt.close(fig)

    # LaTeX
    tex_path = os.path.join(out_dir, "boxplot.tex")
    save_latex_snippet(
        out_tex=tex_path,
        png_name="boxplot.png",
        title=f"Objective distribution across 30 seeds ({os.path.basename(base_dir)})",
    )

    print("=== boxplot ===")
    print(f"base_dir : {base_dir}")
    print(f"out_dir  : {out_dir}")
    print(f"saved    : {png_path}")
    print(f"saved    : {tex_path}")
    print(f"saved    : {stats_path}")

if __name__ == "__main__":
    main()
