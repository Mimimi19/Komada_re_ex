# -*- coding: utf-8 -*-
"""
src/tools/boxplot_compare.py

複数 ROI の band_full の seed_* / correlation.txt を読み込み，
横に並んだ箱ひげ図として比較し PDF 保存する。

実行例:
    uv run python src/tools/boxplot_compare.py \
        --base scripts/limit \
        --roi "1,2,13,14"
        
        
    uv run python src/tools/boxplot_compare.py \
        --base scripts/spring04/scripts/limit\
        --roi "1,2,3,4,5,6,7,8,9,10,11,12,13,14" \
"""

import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D  # ★ 凡例用のダミーオブジェクトに使う

try:
    import japanize_matplotlib
except ImportError:
    pass



# ==========================================================
# ROI → ラベル（グラフタイトルなどに使う）
# ==========================================================
# ROI_LABELS = {
#     1:  "ROI 1  CBC1 (OFF)",
#     2:  "ROI 2  CBC2 (OFF)",
#     3:  "ROI 3  CBC3a (OFF)",
#     4:  "ROI 4  CBC3b (OFF)",
#     5:  "ROI 5  CBC4 (OFF)",
#     6:  "ROI 6  CBC5t (ON)",
#     7:  "ROI 7  CBC5o (ON)",
#     8:  "ROI 8  CBC5i (ON)",
#     9:  "ROI 9  CBCX (ON)",
#     10: "ROI10 CBC6 (ON)",
#     11: "ROI11 CBC7 (ON)",
#     12: "ROI12 CBC8 (ON)",
#     13: "ROI13 CBC9 (ON)",
#     14: "ROI14 RBC (ON)",
# }

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

# -------------------------------
# correlation.txt の読み込み
# -------------------------------
def _read_float(path: str):
    try:
        s = open(path, "r", encoding="utf-8").read().strip()
        for tok in s.replace(",", " ").split():
            try:
                return float(tok)
            except Exception:
                pass
        return float(s)
    except Exception:
        return None


def read_roi_correlations(path_band_full: str):
    """band_full 配下から correlation を読み込む"""
    corrs = []
    seed_ok = []
    seed_err = []

    seed_dirs = sorted(glob.glob(os.path.join(path_band_full, "seed_*")))
    for sd in seed_dirs:
        sname = os.path.basename(sd)
        cpath = os.path.join(sd, "correlation.txt")
        if not os.path.exists(cpath):
            seed_err.append((sname, "missing"))
            continue

        v = _read_float(cpath)
        if v is None or not np.isfinite(v):
            seed_err.append((sname, "parse_error"))
            continue

        corrs.append(v)
        seed_ok.append(sname)

    return np.array(corrs, float), seed_ok, seed_err


# -------------------------------
# LaTeX snippet
# -------------------------------
def save_latex_snippet(out_tex, pdf_name, title):
    tex = f"""\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=0.95\\linewidth]{{{pdf_name}}}
  \\caption{{{title}}}
  \\label{{fig:{os.path.splitext(pdf_name)[0]}}}
\\end{{figure}}
"""
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)


# -------------------------------
# main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="例: scripts/limit")
    ap.add_argument("--roi", required=True,
                    help="例: \"1,2,13,14\"")
    ap.add_argument("--objective", default="band_full",
                    help="使用する目的関数ディレクトリ (default: band_full)")
    args = ap.parse_args()

    base_dir = args.base
    roi_list = [int(x) for x in args.roi.replace(" ", "").split(",") if x != ""]
    objective = args.objective

    out_dir = os.path.join(base_dir, "boxplot_compare")
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------
    # ROIごとに correlation を読み込み
    # -------------------------------
    roi_corrs = []
    roi_labels = []
    roi_errs = {}

    for roi in roi_list:
        path = os.path.join(base_dir, f"roi_{roi}", objective)
        if not os.path.isdir(path):
            print(f"[WARN] directory not found: {path}")
            continue

        corr, seeds_ok, seeds_err = read_roi_correlations(path)
        roi_corrs.append(corr)
        roi_labels.append(ROI_LABELS.get(roi, f"ROI {roi}"))
        roi_errs[roi] = seeds_err

        print(f"[OK] ROI {roi}: {len(corr)} values, {len(seeds_err)} errors")

    if len(roi_corrs) == 0:
        print("ERROR: 有効な ROI がありません。")
        return

    # -------------------------------
    # PDF プロット
    # -------------------------------
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # showmeans=True なので「緑三角 = 平均」「白丸 = 外れ値」が描かれる
    bp = ax.boxplot(roi_corrs, labels=roi_labels, showmeans=True, vert=False)
    ax.set_title(f"Distribution of Post-optimization Correlation Coefficients Across BC Subtypes", fontsize=14)
    ax.set_xlabel("spearman correlation", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)

    # --- 凡例（緑三角=平均, 白丸=外れ値）を明示 ---
    mean_proxy = Line2D(
        [], [], marker="^", color="green", linestyle="None",
        markersize=8, label="mean"
    )
    outlier_proxy = Line2D(
        [], [], marker="o", color="black", linestyle="None",
        markerfacecolor="white", markersize=6, label="outlier"
    )
    ax.legend(handles=[mean_proxy, outlier_proxy], loc="upper left", fontsize=10)
    ax.tick_params(labelsize=10)

    pdf_name = "compare_roi_" + "_".join([str(r) for r in roi_list]) + ".pdf"
    pdf_path = os.path.join(out_dir, pdf_name)

    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close(fig)

    # -------------------------------
    # 数値のまとめ
    # -------------------------------
    stat_path = os.path.join(out_dir, "stats_compare.txt")
    with open(stat_path, "w", encoding="utf-8") as f:
        f.write(f"objective={objective}\n")
        f.write(f"ROI list={roi_list}\n\n")

        for roi, c in zip(roi_list, roi_corrs):
            if len(c) == 0:
                f.write(f"ROI {roi}: no data\n\n")
                continue

            f.write(f"[ROI {roi}]\n")
            f.write(f"n = {len(c)}\n")
            f.write(f"mean = {np.mean(c):.6f}\n")
            f.write(f"median = {np.median(c):.6f}\n")
            f.write(f"max = {np.max(c):.6f}\n")
            f.write(f"min = {np.min(c):.6f}\n")
            f.write(f"errors = {roi_errs.get(roi)}\n\n")

    # -------------------------------
    # LaTeX snippet 出力
    # -------------------------------
    tex_path = os.path.join(out_dir, pdf_name.replace(".pdf", ".tex"))
    save_latex_snippet(tex_path, pdf_name,
                       f"ROI comparison ({objective})")

    print("\n=== boxplot_compare DONE ===")
    print("out_dir :", out_dir)
    print("saved   :", pdf_path)
    print("saved   :", tex_path)
    print("saved   :", stat_path)


if __name__ == "__main__":
    main()
