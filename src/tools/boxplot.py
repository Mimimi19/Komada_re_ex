# -*- coding: utf-8 -*-
"""
src/tools/boxplot.py

- scripts/limit/roi_1/band_low_only/seed_01/correlation.txt のようなファイルを集計
- objective（band_low_only 等）ごとに seed 分布を箱ひげ図で可視化
- PNG と LaTeX で貼れる figure .tex を出力

使い方:
  uv run python src/tools/boxplot.py
  uv run python src/tools/boxplot.py --base scripts/limit/roi_1/band_full
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

    Returns
    -------
    corr : np.ndarray
        読み取り成功した objective 値（seed 順）
    seeds_ok : list[str]
        corr と同じ順番の seed ディレクトリ名（例: 'seed_01'）
    seeds_err : list[tuple[str,str]]
        読み取りに失敗した seed と理由のリスト (seed_name, reason)
    """
    corr = []
    seeds_ok = []
    seeds_err = []

    seed_dirs = sorted(glob.glob(os.path.join(base_dir, "seed_*")))
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        cpath = os.path.join(seed_dir, "correlation.txt")
        if not os.path.exists(cpath):
            seeds_err.append((seed_name, "missing_correlation.txt"))
            continue

        v = _read_float(cpath)
        if v is None:
            seeds_err.append((seed_name, "parse_error"))
            continue
        if not np.isfinite(v):
            seeds_err.append((seed_name, "non_finite"))
            continue

        corr.append(float(v))
        seeds_ok.append(seed_name)

    return np.array(corr, dtype=float), seeds_ok, seeds_err

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
    DEFAULT_BASE = "scripts/limit_full/roi_14"
    OUT_SUBDIR = "boxplot"
    # ==========================================================

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=DEFAULT_BASE, help="seed_*/correlation.txt が入ったディレクトリ")
    ap.add_argument("--success-min", type=float, default=0.1, help="成功とみなす objective の下限（これ以下は失敗扱い）")
    args = ap.parse_args()

    base_dir = args.base
    if not os.path.isdir(base_dir):
        raise SystemExit(f"[Error] base_dir not found: {base_dir}")

    corr, seeds_ok, seeds_err = collect_correlations(base_dir)
    if corr.size == 0:
        raise SystemExit("[Error] No correlation.txt found under seed_*")
    out_dir = os.path.join(base_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    best = float(np.max(corr))
    med  = float(np.median(corr))
    p95  = float(np.quantile(corr, 0.95))
    mean = float(np.mean(corr))

    # 成功 seed の分離（成功率と成功分布を別指標として出す）
    success_min = float(args.success_min)
    success_mask = corr > success_min
    corr_s = corr[success_mask]

    n_total_dirs = len(sorted(glob.glob(os.path.join(base_dir, "seed_*"))))
    n_valid = int(corr.size)
    n_err = int(len(seeds_err))

    # valid の中で「失敗」（成功閾値未満）になった seed
    fail_seeds = [s for s, v in zip(seeds_ok, corr) if v <= success_min]
    success_seeds = [s for s, v in zip(seeds_ok, corr) if v > success_min]

    n_success = int(len(success_seeds))
    n_fail = int(len(fail_seeds))
    success_rate = (n_success / n_valid) if n_valid else 0.0

    if n_success > 0:
        best_s = float(np.max(corr_s))
        med_s  = float(np.median(corr_s))
        p95_s  = float(np.quantile(corr_s, 0.95))
        mean_s = float(np.mean(corr_s))
    else:
        best_s = med_s = p95_s = mean_s = float("nan")

    # 保存（数値）
    stats_path = os.path.join(out_dir, "summary_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        # 全体（失敗混在: 読み取り成功分のみ）
        f.write(f"n_total_seed_dirs={n_total_dirs}\n")
        f.write(f"n_valid={n_valid}\n")
        f.write(f"n_err={n_err}\n")
        f.write(f"best_valid={best:.6f}\n")
        f.write(f"mean_valid={mean:.6f}\n")
        f.write(f"median_valid={med:.6f}\n")
        f.write(f"p95_valid={p95:.6f}\n")
        f.write("\n")
        # 成功率と成功分布（ここが分離報告の本体）
        f.write(f"success_min={success_min:.6f}\n")
        f.write(f"n_success={n_success}\n")
        f.write(f"n_fail={n_fail}\n")
        f.write(f"success_rate={success_rate:.6f}\n")
        f.write(f"best_success={best_s:.6f}\n")
        f.write(f"mean_success={mean_s:.6f}\n")
        f.write(f"median_success={med_s:.6f}\n")
        f.write(f"p95_success={p95_s:.6f}\n")
        f.write("\n")

        # どの seed で問題が起きたか（要求事項）
        if seeds_err:
            f.write("[read_errors]")
            for seed_name, reason in seeds_err:
                f.write(f"{seed_name}	{reason}\n")
            f.write("\n")

        if fail_seeds:
            f.write("[below_success_min]\n")
            for seed_name in fail_seeds:
                f.write(f"{seed_name}\n")
            f.write("\n")

    # プロット
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.boxplot(corr, vert=True, showmeans=True)
    ax.set_title("最適化後の相関係数の分布", fontsize=14)
    ax.set_ylabel("最適化後の相関係数")
    ax.set_xticks([1])
    ax.set_xticklabels([os.path.basename(base_dir)])

    txt = (
        f"valid: n={n_valid}  median={med:.3f}  p95={p95:.3f}\n"
        f"success: {n_success}/{n_valid}  median={med_s:.3f}"
    )
    ax.text(0.02, 0.02, txt, transform=ax.transAxes)

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, "boxplot.png")
    pdf_path = os.path.join(out_dir, "boxplot_compare.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

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
    print(f"saved    : {pdf_path}")
    print(f"saved    : {tex_path}")
    print(f"saved    : {stats_path}")

if __name__ == "__main__":
    main()
