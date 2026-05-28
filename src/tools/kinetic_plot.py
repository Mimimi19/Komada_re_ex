# src/tools/kinetic_plot.py
# -*- coding: utf-8 -*-
"""

使い方:
    # ROI 1〜14 をまとめて処理
    uv run python src/tools/kinetic_plot.py --base scripts/spring04/scripts/limit --roi "1,2,3,4,5,6,7,8,9,10,11,12,13,14"

    # 特定 ROI だけ処理したい場合
    uv run python src/tools/kinetic_plot.py scripts/limit --roi 11

    # 目的関数ディレクトリを変える場合
    uv run python src/tools/kinetic_plot.py scripts/limit --objective band_low_only
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

# ----------------------------------------------------------
# ROI → ラベル
# ----------------------------------------------------------
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

ROI_COLORS = {
    # OFF: 寒色系
    1:  "navy",
    2:  "blue",
    3:  "royalblue",
    4:  "deepskyblue",
    5:  "cyan",

    # ON: 暖色系
    6:  "darkred",
    7:  "red",
    8:  "orangered",
    9:  "darkorange",
    10: "orange",
    11: "coral",
    12: "tomato",
    13: "salmon",

    # Rod bipolar
    14: "black",
}
def add_subtype_legend(ax, roi_list: list[int], best_params_list: list[dict], loc="upper left"):
    """
    グラフ内にサブタイプ色凡例を表示する。
    best_params が None の ROI は凡例から除外する。
    """
    handles = []

    for roi, p in zip(roi_list, best_params_list):
        if p is None:
            continue

        label = ROI_LABELS.get(roi, f"ROI {roi}")
        color = ROI_COLORS.get(roi, "gray")

        handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=7,
            label=label,
        )
        handles.append(handle)

    ax.legend(
        handles=handles,
        loc=loc,
        fontsize=7,
        frameon=True,
        framealpha=0.85,
        borderpad=0.6,
        labelspacing=0.4,
        handletextpad=0.5,
    )

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


def read_roi_corr(path_band_full: str):
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

def sort_by_corr(corrs: np.ndarray, seeds_ok: list[int]):
    """correlation の値で seeds_ok をソートする"""
    if len(corrs) != len(seeds_ok):
        raise ValueError("corrs と seeds_ok は同じ長さでなければなりません")
    sorted_indices = np.argsort(corrs)[::-1]  # 降順ソート
    sorted_corrs = corrs[sorted_indices]
    sorted_seeds = [seeds_ok[i] for i in sorted_indices]
    return sorted_corrs, sorted_seeds

def _read_param_file(path: str):
    """ka.txt などのパラメータファイルから値を読み込む"""
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

def collect_kinetics_and_best(base_dir: str, best_seed_keys: list[str]):
    """

    """
    keys = ["ka", "kfi", "kfr", "ksi", "ksr"]
    vals = {k: [] for k in keys}
    seeds_ok: list[int] = []

    best_seed = None
    best_corr = None
    best_params = None

    seed_dirs = sorted(glob.glob(os.path.join(base_dir, "seed_*")))
    for sd in seed_dirs:
        if os.path.basename(sd) == best_seed_keys:
            
            base = os.path.basename(sd)
            # seed_XX の XX 部分だけ int で取り出す
            try:
                seed_num = int(base.split("_")[1])
            except Exception:
                continue

            # Kinetics パラメータ読み込み
            pvals = {}
            missing_param = False
            for k in keys:
                v = _read_param_file(os.path.join(sd, f"{k}.txt"))
                if v is None or not np.isfinite(v):
                    missing_param = True
                    break
                pvals[k] = float(v)

            if missing_param:
                continue

            # ベスト seed として保存
            if best_corr is None or (pvals.get("correlation", 0) ):
                best_seed = seed_num
                best_corr = pvals.get("correlation", 0)
                best_params = pvals

    # numpy 配列に変換
    data = {k: np.asarray(v, dtype=float) for k, v in vals.items()}

    return best_params



def save_fast_ratio_scatter(
    out_dir: str,
    roi_list: list[int],
    best_params_list: list[dict]
):
    """
    kfi/ka vs kfr/ka の散布図を保存
    """

    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    for roi, p in zip(roi_list, best_params_list):

        if p is None:
            continue

        ka = p["ka"]

        x = p["kfi"] / ka
        y = p["kfr"] / ka

        label = ROI_LABELS.get(roi, f"ROI {roi}")
        color = ROI_COLORS.get(roi, "gray")

        ax.scatter(
            x,
            y,
            s=90,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3
        )

        ax.text(
            x,
            y,
            f" {label}",
            fontsize=8,
            color=color,
            ha="left",
            va="center"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$k_{fi} / k_a$", fontsize=14)
    ax.set_ylabel(r"$k_{fr} / k_a$", fontsize=14)

    ax.set_title(r"Fast kinetics ratio scatter plot ($k_{fi}/k_a$ vs $k_{fr}/k_a$)", fontsize=18)

    ax.grid(True, which="both", alpha=0.3)

    # -----------------------------------
    # y = x reference line
    # -----------------------------------
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    line_min = min(xmin, ymin)
    line_max = max(xmax, ymax)

    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        linewidth=1.2,
        color="black",
        alpha=0.7,
        zorder=1
    )
    add_subtype_legend(ax, roi_list, best_params_list, loc="upper left")

    plt.tight_layout()

    save_path = os.path.join(
        out_dir,
        "kfi_kfr_ratio_scatter.pdf"
    )

    plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"[SAVE] {save_path}")

def save_slow_ratio_scatter(
    out_dir: str,
    roi_list: list[int],
    best_params_list: list[dict]
):
    """
    ksi/ka vs ksr/ka の散布図を保存
    """

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for roi, p in zip(roi_list, best_params_list):

        if p is None:
            continue

        ka = p["ka"]

        x = p["ksi"] / ka
        y = p["ksr"] / ka

        label = ROI_LABELS.get(roi, f"ROI {roi}")
        color = ROI_COLORS.get(roi, "gray")

        ax.scatter(
            x,
            y,
            s=90,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3
        )

        ax.text(
            x,
            y,
            f" {label}",
            fontsize=8,
            color=color,
            ha="left",
            va="center"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$k_{si} / k_a$", fontsize=14)
    ax.set_ylabel(r"$k_{sr} / k_a$", fontsize=14)

    ax.set_title(r"Slow kinetics ratio scatter plot ($k_{si}/k_a$ vs $k_{sr}/k_a$)", fontsize=18)

    ax.grid(True, which="both", alpha=0.3)
    # -----------------------------------
    # y = x reference line
    # -----------------------------------
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    line_min = min(xmin, ymin)
    line_max = max(xmax, ymax)

    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        linewidth=1.2,
        color="black",
        alpha=0.7,
        zorder=1
    )
    add_subtype_legend(ax, roi_list, best_params_list, loc="upper left")
    plt.tight_layout()

    save_path = os.path.join(
        out_dir,
        "ksi_ksr_ratio_scatter.pdf"
    )

    plt.savefig(
        save_path,
        bbox_inches="tight",
        format="pdf"
    )

    plt.close()

    print(f"[SAVE] {save_path}")





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

    out_dir = os.path.join(base_dir, "kinetic_plot")
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------
    # ROIごとに correlation を読み込み
    # -------------------------------
    roi_corrs = []
    roi_labels = []
    roi_errs = {}
    sorted_seeds = []
    sorted_corrs = []
    best_seeds = []
    best_params_list = []
    for roi in roi_list:
        path = os.path.join(base_dir, f"roi_{roi}", objective)
        if not os.path.isdir(path):
            print(f"[WARN] directory not found: {path}")
            continue

        corr, seeds_ok, seeds_err = read_roi_corr(path)
        roi_corrs.append(corr)
        roi_labels.append(ROI_LABELS.get(roi, f"ROI {roi}"))
        scorr, sseeds = sort_by_corr(corr, seeds_ok) 
        sorted_corrs.append(scorr)
        sorted_seeds.append(sseeds) 
        roi_errs[roi] = seeds_err
        
        best_seeds.append(sseeds[0] if len(sseeds) > 0 else None)
        
        best_param = collect_kinetics_and_best(path, sseeds[0] if len(sseeds) > 0 else None)
        print(f"Best seed for ROI {roi}: {sseeds[0] if len(sseeds) > 0 else 'N/A'}, params: {best_param}")
        best_params_list.append(best_param)
        

        print(f"[OK] ROI {roi}: {len(corr)} values, {len(seeds_err)} errors")

    if len(roi_corrs) == 0:
        print("ERROR: 有効な ROI がありません。")
        return
    
    save_fast_ratio_scatter(
        out_dir,
        roi_list,
        best_params_list
    )

    save_slow_ratio_scatter(
        out_dir,
        roi_list,
        best_params_list
    )
    
    

    
if __name__ == "__main__":
    main()
