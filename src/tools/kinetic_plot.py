# src/tools/kinetic_plot.py
# -*- coding: utf-8 -*-
"""
Kinetics パラメータ (ka, kfi, kfr, ksi, ksr) の分布と相関を可視化するツール。

想定ディレクトリ構造:
    <base_dir>/
      seed_01/
        ka.txt
        kfi.txt
        kfr.txt
        ksi.txt
        ksr.txt
      seed_02/
        ...
      ...

出力:
    <base_dir>/kinetic_plot/
        kinetic_linear.pdf          # 線形スケールの箱ひげ図
        kinetic_log.pdf             # logスケールの箱ひげ図
        kinetic_scatter_matrix.pdf  # パラメータ相関の散布図マトリクス
        kinetic_stats.txt           # 平均・中央値などの統計
        kinetic_linear.tex          # LaTeX 用 figure snippet

実行例:
    uv run python src/tools/kinetic_plot.py \
      --base scripts/limit/roi_1/band_full
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 日本語フォント（ある場合のみ）
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


PARAM_NAMES = ["ka", "kfi", "kfr", "ksi", "ksr"]


def infer_roi_label(base_dir: str) -> str:
    """
    base_dir から ROI ラベルを推定する。
    例:
        scripts/limit/roi_1/band_full -> "ROI 1"
        scripts/limit/roi_14/band_full -> "ROI 14"
    見つからなければ空文字を返す。
    """
    parts = re.split(r"[\\/]", base_dir)
    for p in parts:
        m = re.match(r"roi_(\d+)", p)
        if m:
            return f"ROI {m.group(1)}"
    return ""


def read_param_file(path: str):
    """テキストから float を読む。失敗したら None。"""
    if not os.path.exists(path):
        return None
    try:
        s = open(path, "r", encoding="utf-8").read().strip()
        # "ka: 0.123" みたいな形式にも一応対応
        for tok in s.replace(",", " ").split():
            try:
                v = float(tok)
                if np.isfinite(v):
                    return v
            except Exception:
                continue
        v = float(s)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def collect_kinetic_params(base_dir: str):
    """
    base_dir/seed_*/ から ka, kfi, kfr, ksi, ksr を読み込む。

    Returns
    -------
    data_dict : dict[str, np.ndarray]
        各パラメータ名 -> 値の配列（seed順）
    seeds_ok : list[str]
        有効な seed_* 名
    seeds_err : list[tuple[str,str]]
        読み込み失敗 seed と理由
    """
    data = {p: [] for p in PARAM_NAMES}
    seeds_ok = []
    seeds_err = []

    seed_dirs = sorted(glob.glob(os.path.join(base_dir, "seed_*")))
    if len(seed_dirs) == 0:
        raise RuntimeError(f"seed_* が見つかりません: {base_dir}")

    for sd in seed_dirs:
        seed_name = os.path.basename(sd)
        vals = {}
        ok = True
        reason = ""

        for p in PARAM_NAMES:
            path = os.path.join(sd, f"{p}.txt")
            v = read_param_file(path)
            if v is None:
                ok = False
                reason = f"{p}.txt missing or invalid"
                break
            vals[p] = v

        if not ok:
            seeds_err.append((seed_name, reason))
            continue

        for p in PARAM_NAMES:
            data[p].append(vals[p])
        seeds_ok.append(seed_name)

    for p in PARAM_NAMES:
        data[p] = np.asarray(data[p], dtype=float)

    return data, seeds_ok, seeds_err


def save_latex_snippet(out_tex: str, pdf_name: str, title: str):
    """LaTeX の \includegraphics で貼れる figure snippet を保存"""
    tex = f"""\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=0.95\\linewidth]{{{pdf_name}}}
  \\caption{{{title}}}
  \\label{{fig:{os.path.splitext(pdf_name)[0]}}}
\\end{{figure}}
"""
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)


def plot_boxplots(data: dict, out_dir: str, base_name: str, roi_label: str):
    """線形 & log の箱ひげ図を保存"""
    values = [data[p] for p in PARAM_NAMES]

    if roi_label:
        title_lin = f"{roi_label}: Kinetics パラメータ分布（線形スケール）"
        title_log = f"{roi_label}: Kinetics パラメータ分布（対数スケール）"
        caption_lin = f"{roi_label} における kinetics パラメータ分布（線形スケール）"
    else:
        title_lin = "Kinetics パラメータ分布（線形スケール）"
        title_log = "Kinetics パラメータ分布（対数スケール）"
        caption_lin = "Kinetics パラメータ分布（線形スケール）"

    # ---- 線形スケール ----
    fig_lin = plt.figure(figsize=(8, 5))
    ax_lin = fig_lin.add_subplot(1, 1, 1)
    ax_lin.boxplot(values, tick_labels=PARAM_NAMES, showmeans=True, vert=True)
    ax_lin.set_title(title_lin, fontsize=14)
    ax_lin.set_ylabel("Parameter value")
    ax_lin.grid(True, linestyle="--", alpha=0.3)

    # 凡例（緑三角=平均, 白丸=外れ値）
    mean_proxy = Line2D(
        [], [], marker="^", color="green", linestyle="None",
        markersize=8, label="平均値 (mean)"
    )
    outlier_proxy = Line2D(
        [], [], marker="o", color="black", linestyle="None",
        markerfacecolor="white", markersize=6, label="外れ値 (outlier)"
    )
    ax_lin.legend(handles=[mean_proxy, outlier_proxy], loc="upper right")

    fig_lin.tight_layout()
    pdf_linear = os.path.join(out_dir, f"{base_name}_linear.pdf")
    fig_lin.savefig(pdf_linear, bbox_inches="tight")
    plt.close(fig_lin)

    # ---- logスケール（値が正のものだけを想定）----
    fig_log = plt.figure(figsize=(8, 5))
    ax_log = fig_log.add_subplot(1, 1, 1)
    ax_log.boxplot(values, tick_labels=PARAM_NAMES, showmeans=True, vert=True)
    ax_log.set_yscale("log")
    ax_log.set_title(title_log, fontsize=14)
    ax_log.set_ylabel("Parameter value (log scale)")
    ax_log.grid(True, linestyle="--", alpha=0.3)
    fig_log.tight_layout()

    pdf_log = os.path.join(out_dir, f"{base_name}_log.pdf")
    fig_log.savefig(pdf_log, bbox_inches="tight")
    plt.close(fig_log)

    # LaTeX snippet（線形を採用）
    tex_path = os.path.join(out_dir, f"{base_name}_linear.tex")
    save_latex_snippet(
        tex_path,
        os.path.basename(pdf_linear),
        title=caption_lin,
    )

    return pdf_linear, pdf_log, tex_path


def plot_scatter_matrix(data: dict, out_dir: str, base_name: str, roi_label: str):
    """パラメータ間の相関を見るための散布図マトリクスを保存"""
    n_param = len(PARAM_NAMES)
    fig, axes = plt.subplots(n_param, n_param, figsize=(3 * n_param, 3 * n_param))

    for i, pi in enumerate(PARAM_NAMES):
        xi = data[pi]
        for j, pj in enumerate(PARAM_NAMES):
            ax = axes[i, j]
            yj = data[pj]

            if i == j:
                # 対角成分: ヒストグラム
                ax.hist(xi, bins=15, alpha=0.7)
            else:
                ax.scatter(yj, xi, s=10, alpha=0.5)

            if i == n_param - 1:
                ax.set_xlabel(pj)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(pi)
            else:
                ax.set_yticklabels([])

    title = "Kinetics パラメータ相関マトリクス"
    if roi_label:
        title = f"{roi_label}: " + title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = os.path.join(out_dir, f"{base_name}_scatter_matrix.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return pdf_path


def save_stats(data: dict, seeds_ok, seeds_err, out_dir: str, base_name: str, roi_label: str):
    """平均・中央値などの統計情報をテキスト保存"""
    path = os.path.join(out_dir, f"{base_name}_stats.txt")
    with open(path, "w", encoding="utf-8") as f:
        if roi_label:
            f.write(f"{roi_label}\n")
            f.write("-" * len(roi_label) + "\n\n")

        f.write(f"n_valid_seeds = {len(seeds_ok)}\n")
        f.write(f"n_error_seeds = {len(seeds_err)}\n\n")
        f.write("[valid_seeds]\n")
        for sname in seeds_ok:
            f.write(f"{sname}\n")
        f.write("\n")

        if seeds_err:
            f.write("[error_seeds]\n")
            for sname, reason in seeds_err:
                f.write(f"{sname}\t{reason}\n")
            f.write("\n")

        for p in PARAM_NAMES:
            arr = np.asarray(data[p], float)
            if arr.size == 0:
                continue
            f.write(f"[{p}]\n")
            f.write(f"  n      = {arr.size}\n")
            f.write(f"  mean   = {np.mean(arr):.6g}\n")
            f.write(f"  median = {np.median(arr):.6g}\n")
            f.write(f"  std    = {np.std(arr):.6g}\n")
            f.write(f"  min    = {np.min(arr):.6g}\n")
            f.write(f"  max    = {np.max(arr):.6g}\n\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        required=True,
        help="例: scripts/limit/roi_1/band_full",
    )
    args = ap.parse_args()

    base_dir = args.base
    if not os.path.isdir(base_dir):
        raise SystemExit(f"[Error] base_dir not found: {base_dir}")

    # ROI ラベル推定
    roi_label = infer_roi_label(base_dir)

    # パラメータ収集
    data, seeds_ok, seeds_err = collect_kinetic_params(base_dir)
    if len(seeds_ok) == 0:
        raise SystemExit("[Error] 有効な seed_* がありません（パラメータ欠損）")

    out_dir = os.path.join(base_dir, "kinetic_plot")
    os.makedirs(out_dir, exist_ok=True)

    base_name = "kinetic"

    # 箱ひげ図
    pdf_lin, pdf_log, tex_path = plot_boxplots(data, out_dir, base_name, roi_label)

    # 散布図マトリクス
    pdf_scatter = plot_scatter_matrix(data, out_dir, base_name, roi_label)

    # 統計
    stats_path = save_stats(data, seeds_ok, seeds_err, out_dir, base_name, roi_label)

    print("=== kinetic_plot DONE ===")
    print(f"base_dir : {base_dir}")
    print(f"out_dir  : {out_dir}")
    print(f"ROI      : {roi_label if roi_label else '(unknown)'}")
    print(f"boxplot_linear : {pdf_lin}")
    print(f"boxplot_log    : {pdf_log}")
    print(f"scatter_matrix : {pdf_scatter}")
    print(f"stats          : {stats_path}")
    print(f"latex          : {tex_path}")


if __name__ == "__main__":
    main()
