#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/tools/boxplot_compare.py

同一 ROI 内の複数 objective（例: band_low_only / band_main_only）の
最終スコア(correlation.txt) を集計して、箱ひげ図を横並びで描画します。

- low / main がどちらか分かるようにラベル表示
- 薄い破線グリッドを追加（値が読みやすい）
- PNG と LaTeX(.tex) を出力

使い方:
  uv run python src/tools/boxplot_compare.py --roi-dir scripts/limit/roi_1
  uv run python src/tools/boxplot_compare.py --roi-dir scripts/limit/roi_1 --objectives band_low_only,band_main_only

出力:
  scripts/limit/roi_1/boxplot_compare/boxplot_compare.png
  scripts/limit/roi_1/boxplot_compare/boxplot_compare.tex
  scripts/limit/roi_1/boxplot_compare/summary_stats.tsv
"""
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント対応
try:
    import japanize_matplotlib
except ImportError:
    pass


def _read_scores(obj_dir: str):
    """obj_dir/seed_*/correlation.txt を読み、float 配列で返す"""
    pattern = os.path.join(obj_dir, "seed_*", "correlation.txt")
    files = sorted(glob.glob(pattern))
    vals = []
    for fp in files:
        try:
            v = float(np.loadtxt(fp))
            # correlation.txt は「最大化したい相関係数」を保存している前提
            vals.append(v)
        except Exception:
            pass
    return np.array(vals, dtype=float), files

def _stats(x: np.ndarray):
    if x.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, q05=np.nan, q50=np.nan, q95=np.nan, best=np.nan)
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        std=float(np.std(x)),
        q05=float(np.quantile(x, 0.05)),
        q50=float(np.quantile(x, 0.50)),
        q95=float(np.quantile(x, 0.95)),
        best=float(np.max(x)),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-dir", required=True, help="例: scripts/limit/roi_1")
    ap.add_argument("--objectives", default="band_low_only,band_main_only",
                    help="カンマ区切り（例: band_low_only,band_main_only,band_full）")
    ap.add_argument("--outdir", default="", help="出力先（空なら roi-dir/boxplot_compare）")
    args = ap.parse_args()

    roi_dir = args.roi_dir.rstrip("/")
    objectives = [s.strip() for s in args.objectives.split(",") if s.strip()]
    if not objectives:
        raise SystemExit("No objectives given.")

    outdir = args.outdir.strip() or os.path.join(roi_dir, "boxplot_compare")
    os.makedirs(outdir, exist_ok=True)

    # 収集
    data = []
    labels = []
    stats_rows = []
    for obj in objectives:
        obj_dir = os.path.join(roi_dir, obj)
        scores, files = _read_scores(obj_dir)
        data.append(scores)
        labels.append(obj)

        st = _stats(scores)
        stats_rows.append((obj, st["n"], st["best"], st["q95"], st["q50"], st["q05"], st["mean"], st["std"]))

        print(f"[{obj}] n={st['n']} best={st['best']:.6f} q95={st['q95']:.6f} median={st['q50']:.6f}")

    # TSV保存（論文用にコピペしやすい）
    tsv_path = os.path.join(outdir, "summary_stats.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("objective\tn\tbest\tq95\tmedian\tq05\tmean\tstd\n")
        for row in stats_rows:
            f.write("\t".join([str(x) for x in row]) + "\n")

    # プロット
    plt.figure(figsize=(max(6, 1.8 * len(objectives)), 5.2))
    ax = plt.gca()

    # 箱ひげ図（横並び）
    positions = np.arange(1, len(objectives) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    # 色指定はしない（要望: 色固定しない）。ただし塗りは薄い灰で統一して可読性を確保
    for b in bp["boxes"]:
        b.set_alpha(0.35)

    ax.set_xticks(positions)
    # 低/主が分かるようにラベル（そのまま objective 名）
    ax.set_xticklabels(labels, rotation=0)

    ax.set_title(os.path.basename(roi_dir) + "における周波数帯ごとの最適化結果比較")
    ax.set_ylabel("最適化後のスピアマンの相関係数")

    # 読みやすいグリッド：薄い破線
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    # 目盛りも少し増やす（グリッドが見やすい）
    ax.minorticks_on()
    ax.grid(True, which="minor", axis="y", linestyle="--", linewidth=0.5, alpha=0.18)

    # median 値を上に表示（数値が分かりづらい問題に対処）
    for i, scores in enumerate(data, start=1):
        if scores.size == 0:
            continue
        med = float(np.quantile(scores, 0.5))
        ax.text(i, med, f"{med:.3f}", ha="center", va="bottom", fontsize=9)

    pdf_path = os.path.join(outdir, "boxplot_compare.pdf")
    plt.tight_layout()
    pdf_path = os.path.join(outdir, "boxplot_compare.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()

    # LaTeX snippet
    tex_path = os.path.join(outdir, "boxplot_compare.tex")
    tex = r"""\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{%s}
    \caption{Objective comparison (boxplots) for %s.}
    \label{fig:%s_objective_boxplot}
    \end{figure}
     """ % (os.path.basename(pdf_path), os.path.basename(roi_dir), os.path.basename(roi_dir))

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    print("\nSaved:")
    print("  " + pdf_path)
    print("  " + tex_path)
    print("  " + tsv_path)

if __name__ == "__main__":
    main()
