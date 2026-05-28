# src/tools/nonlinear_scatter.py
# -*- coding: utf-8 -*-
"""
Nonlinear stage の形状解析ツール。

各 ROI の best seed（correlation 最大）について、非線形関数 u(g) を解析し、
以下の 3 種類の PDF を保存する。

1. nonlinear_midpoint_g.pdf
   - 正規化 u(g) が 0.5 に到達する g の値
   - いわゆる sigmoid の中心位置 / threshold 的な指標

2. nonlinear_transition_width.pdf
   - 正規化 u(g) が low_frac から high_frac に移行するまでの g 幅
   - default は 10% → 90% 幅

3. nonlinear_midpoint_vs_width.pdf
   - 横軸: u=0.5 到達時の g
   - 縦軸: transition width

実行例:
    uv run python src/tools/nonlinear_scatter.py \
        --base scripts/spring04/scripts/limit \
        --roi "1,2,3,4,5,6,7,8,9,10,11,12,13,14"

目的関数ディレクトリを変える場合:
    uv run python src/tools/nonlinear_scatter.py \
        --base scripts/spring04/scripts/limit \
        --roi "1,2,3" \
        --objective band_low_only
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import erf
from matplotlib.lines import Line2D

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


# ==========================================================
# plot settings
# ==========================================================

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


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
    # OFF: cold colors
    1:  "navy",
    2:  "blue",
    3:  "royalblue",
    4:  "deepskyblue",
    5:  "cyan",

    # ON: warm colors
    6:  "darkred",
    7:  "red",
    8:  "orangered",
    9:  "darkorange",
    10: "orange",
    11: "gold",
    12: "tomato",
    13: "salmon",

    # Rod bipolar
    14: "black",
}


# ==========================================================
# utility
# ==========================================================

def _read_float(path: str):
    """txt ファイルから最初に読めた float を返す。"""
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


def read_corr(seed_dir: str):
    p = os.path.join(seed_dir, "correlation.txt")
    if not os.path.exists(p):
        return None

    v = _read_float(p)
    if v is None or not np.isfinite(v):
        return None

    return float(v)


def read_param(seed_dir: str, name: str):
    p = os.path.join(seed_dir, f"{name}.txt")
    if not os.path.exists(p):
        return None

    v = _read_float(p)
    if v is None or not np.isfinite(v):
        return None

    return float(v)


def find_best_seed_dir(roi_objective_dir: str):
    """correlation.txt が最大の seed ディレクトリを返す。"""
    seed_dirs = sorted(glob.glob(os.path.join(roi_objective_dir, "seed_*")))

    best_dir = None
    best_corr = -np.inf

    for sd in seed_dirs:
        corr = read_corr(sd)
        if corr is None:
            continue

        if corr > best_corr:
            best_corr = corr
            best_dir = sd

    if best_dir is None:
        return None, None

    return best_dir, best_corr


def read_nonlinear_params(seed_dir: str):
    """
    N_LNK の主要パラメータを読む。

    想定:
        u(g) = a^(erf(kappa*g + b1) + 1) / ka + b2

    kappa が存在しない古い出力の場合は 1.0 とする。
    """
    a = read_param(seed_dir, "a")
    kappa = read_param(seed_dir, "kappa")
    b1 = read_param(seed_dir, "b1")
    b2 = read_param(seed_dir, "b2")
    ka = read_param(seed_dir, "ka")

    if kappa is None:
        kappa = 1.0

    if any(v is None for v in [a, kappa, b1, b2, ka]):
        return None

    if ka == 0:
        return None

    return {
        "a": float(a),
        "kappa": float(kappa),
        "b1": float(b1),
        "b2": float(b2),
        "ka": float(ka),
    }


def nonlinear_u(g: np.ndarray, params: dict):
    """
    N_LNK.main を直接 import せず、式から u(g) を評価する。
    """
    a = params["a"]
    kappa = params["kappa"]
    b1 = params["b1"]
    b2 = params["b2"]
    ka = params["ka"]

    erf_vec = np.vectorize(erf)
    return (a ** (erf_vec(kappa * g + b1) + 1.0)) / ka + b2


def normalize01(y: np.ndarray):
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None

    if abs(y_max - y_min) < 1e-12:
        return None

    return (y - y_min) / (y_max - y_min)


def crossing_x(g: np.ndarray, y01: np.ndarray, target: float):
    """
    y01 が target に到達する g を線形補間で求める。
    単調増加・単調減少の両方に対応。
    """
    if len(g) != len(y01) or len(g) < 2:
        return None

    # y01 が減少関数なら反転して扱う
    if y01[-1] < y01[0]:
        y_work = 1.0 - y01
        target_work = 1.0 - target
    else:
        y_work = y01
        target_work = target

    diff = y_work - target_work
    hit = np.where(diff[:-1] * diff[1:] <= 0)[0]

    if len(hit) == 0:
        return None

    i = hit[0]

    x0, x1 = g[i], g[i + 1]
    y0, y1 = y_work[i], y_work[i + 1]

    if abs(y1 - y0) < 1e-12:
        return float((x0 + x1) / 2.0)

    return float(x0 + (target_work - y0) * (x1 - x0) / (y1 - y0))


def analyze_nonlinear_params(
    params: dict,
    g_min: float,
    g_max: float,
    n_grid: int,
    low_frac: float,
    high_frac: float,
    mid_frac: float,
):
    """
    非線形関数から midpoint と transition width を計算する。
    """
    g = np.linspace(g_min, g_max, n_grid)
    u = nonlinear_u(g, params)
    u01 = normalize01(u)

    if u01 is None:
        return None

    g_mid = crossing_x(g, u01, mid_frac)
    g_low = crossing_x(g, u01, low_frac)
    g_high = crossing_x(g, u01, high_frac)

    if g_mid is None or g_low is None or g_high is None:
        return None

    return {
        "g_mid": float(g_mid),
        "g_low": float(g_low),
        "g_high": float(g_high),
        "width": float(abs(g_high - g_low)),
    }


def collect_roi_nonlinear_features(
    base_dir: str,
    roi_list: list[int],
    objective: str,
    g_min: float,
    g_max: float,
    n_grid: int,
    low_frac: float,
    high_frac: float,
    mid_frac: float,
):
    rows = []

    for roi in roi_list:
        roi_dir = os.path.join(base_dir, f"roi_{roi}", objective)

        if not os.path.isdir(roi_dir):
            print(f"[WARN] directory not found: {roi_dir}")
            continue

        best_seed_dir, best_corr = find_best_seed_dir(roi_dir)

        if best_seed_dir is None:
            print(f"[WARN] no valid seed found: ROI {roi}")
            continue

        params = read_nonlinear_params(best_seed_dir)

        if params is None:
            print(f"[WARN] nonlinear params not found: {best_seed_dir}")
            continue

        features = analyze_nonlinear_params(
            params=params,
            g_min=g_min,
            g_max=g_max,
            n_grid=n_grid,
            low_frac=low_frac,
            high_frac=high_frac,
            mid_frac=mid_frac,
        )

        if features is None:
            print(f"[WARN] failed to analyze nonlinear curve: ROI {roi}")
            continue

        seed_name = os.path.basename(best_seed_dir)

        row = {
            "roi": roi,
            "label": ROI_LABELS.get(roi, f"ROI {roi}"),
            "color": ROI_COLORS.get(roi, "gray"),
            "seed": seed_name,
            "corr": best_corr,
            **params,
            **features,
        }

        rows.append(row)

        print(
            f"[OK] ROI {roi:02d} {row['label']}: "
            f"{seed_name}, corr={best_corr:.4f}, "
            f"g_mid={row['g_mid']:.4f}, width={row['width']:.4f}"
        )

    return rows

# ==========================================================
# plotting
# ==========================================================

def add_subtype_legend(ax, rows: list[dict], loc="upper left"):
    """
    グラフ内にサブタイプ色凡例を表示する。
    点の近くに文字を置かず、左上にまとめて表示することで重なりを避ける。
    """
    handles = []

    for r in rows:
        handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=r["color"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=7,
            label=r["label"],
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

def save_midpoint_g_plot(rows: list[dict], out_dir: str, mid_frac: float):
    fig, ax = plt.subplots(figsize=(10, 4.8))

    x = np.arange(len(rows))

    for i, r in enumerate(rows):
        ax.scatter(
            i,
            r["g_mid"],
            s=90,
            color=r["color"],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right")

    ax.set_ylabel(r"$g$ at normalized $u(g) = %.2f$" % mid_frac)
    ax.set_title("Nonlinear midpoint position")
    ax.grid(True, axis="y", alpha=0.3)
    add_subtype_legend(ax, rows, loc="upper left")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "nonlinear_midpoint_g.pdf")
    fig.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"[SAVE] {save_path}")


def save_transition_width_plot(
    rows: list[dict],
    out_dir: str,
    low_frac: float,
    high_frac: float,
):
    fig, ax = plt.subplots(figsize=(10, 4.8))

    x = np.arange(len(rows))

    for i, r in enumerate(rows):
        ax.scatter(
            i,
            r["width"],
            s=90,
            color=r["color"],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right")

    ax.set_ylabel(
        r"$g$ width from $u=%.2f$ to $u=%.2f$" % (low_frac, high_frac)
    )
    ax.set_title("Nonlinear transition width")
    ax.grid(True, axis="y", alpha=0.3)
    add_subtype_legend(ax, rows, loc="upper left")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "nonlinear_transition_width.pdf")
    fig.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"[SAVE] {save_path}")


def save_midpoint_vs_width_scatter(rows: list[dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    for r in rows:
        ax.scatter(
            r["g_mid"],
            r["width"],
            s=95,
            color=r["color"],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        ax.text(
            r["g_mid"],
            r["width"],
            " " + r["label"],
            fontsize=8,
            color=r["color"],
            ha="left",
            va="center",
        )

    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)

    ax.set_xlabel(r"$g$ at normalized $u(g)=0.5$", fontsize=14)
    ax.set_ylabel(r"Transition width in $g$", fontsize=14)
    ax.set_title("Nonlinear midpoint vs transition width", fontsize=18)
    ax.grid(True, alpha=0.3)
    add_subtype_legend(ax, rows, loc="upper left")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "nonlinear_midpoint_vs_width.pdf")
    fig.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"[SAVE] {save_path}")


def save_summary_csv(rows: list[dict], out_dir: str):
    save_path = os.path.join(out_dir, "nonlinear_features.csv")

    header = [
        "roi",
        "label",
        "seed",
        "corr",
        "g_mid",
        "g_low",
        "g_high",
        "width",
        "a",
        "kappa",
        "b1",
        "b2",
        "ka",
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")

        for r in rows:
            vals = []
            for h in header:
                v = r[h]
                if isinstance(v, str):
                    vals.append(f'"{v}"')
                else:
                    vals.append(f"{v:.10g}")
            f.write(",".join(vals) + "\n")

    print(f"[SAVE] {save_path}")


# ==========================================================
# main
# ==========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        required=True,
        help="例: scripts/spring04/scripts/limit",
    )

    parser.add_argument(
        "--roi",
        required=True,
        help='例: "1,2,3,4,5,6,7,8,9,10,11,12,13,14"',
    )

    parser.add_argument(
        "--objective",
        default="band_full",
        help="使用する目的関数ディレクトリ。default: band_full",
    )

    parser.add_argument(
        "--g_min",
        type=float,
        default=-16.0,
        help="非線形関数を評価する g の最小値。default: -8.0",
    )

    parser.add_argument(
        "--g_max",
        type=float,
        default=16.0,
        help="非線形関数を評価する g の最大値。default: 8.0",
    )

    parser.add_argument(
        "--n_grid",
        type=int,
        default=20001,
        help="非線形関数評価のグリッド数。default: 20001",
    )

    parser.add_argument(
        "--low_frac",
        type=float,
        default=0.10,
        help="transition width の下側。default: 0.10",
    )

    parser.add_argument(
        "--high_frac",
        type=float,
        default=0.90,
        help="transition width の上側。default: 0.90",
    )

    parser.add_argument(
        "--mid_frac",
        type=float,
        default=0.50,
        help="midpoint として使う u の正規化値。default: 0.50",
    )

    args = parser.parse_args()

    roi_list = [int(x) for x in args.roi.replace(" ", "").split(",") if x != ""]

    out_dir = os.path.join(args.base, "nonlinear_scatter")
    os.makedirs(out_dir, exist_ok=True)

    if not (0.0 < args.low_frac < args.high_frac < 1.0):
        raise ValueError("--low_frac と --high_frac は 0 < low < high < 1 にしてください。")

    if not (0.0 < args.mid_frac < 1.0):
        raise ValueError("--mid_frac は 0 < mid < 1 にしてください。")

    rows = collect_roi_nonlinear_features(
        base_dir=args.base,
        roi_list=roi_list,
        objective=args.objective,
        g_min=args.g_min,
        g_max=args.g_max,
        n_grid=args.n_grid,
        low_frac=args.low_frac,
        high_frac=args.high_frac,
        mid_frac=args.mid_frac,
    )

    if not rows:
        print("ERROR: 有効な nonlinear features がありません。")
        return

    save_midpoint_g_plot(rows, out_dir, args.mid_frac)
    save_transition_width_plot(rows, out_dir, args.low_frac, args.high_frac)
    save_midpoint_vs_width_scatter(rows, out_dir)
    save_summary_csv(rows, out_dir)


if __name__ == "__main__":
    main()
