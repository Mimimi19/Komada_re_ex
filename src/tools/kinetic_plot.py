# src/tools/kinetic_plot.py
# -*- coding: utf-8 -*-
"""
Kinetics パラメータ (ka, kfi, kfr, ksi, ksr) の分布と
相関係数ベースで選んだ「ベスト seed」のパラメータを可視化するツール。

想定ディレクトリ構造:
    <root_dir>/
      roi_1/
        band_full/
          seed_01/
            ka.txt, kfi.txt, kfr.txt, ksi.txt, ksr.txt, correlation.txt, ...
          ...
          seed_30/
            ...
      ...
      roi_14/
        band_full/
          ...

出力:
    <root_dir>/roi_<ROI>/<objective>/kinetic_plot/      （従来と同じ場所）
      kinetic_linear.pdf           ... 各パラメータの箱ひげ図（線形軸）
      kinetic_log.pdf              ... 各パラメータの箱ひげ図（対数軸）
      kinetic_scatter_matrix.pdf   ... パラメータ空間の散布図行列
      kinetic_stats.txt            ... 基本統計量 + best seed 情報
      kinetic_best_seed_bar.pdf    ... ベスト seed のパラメータの棒グラフ

使い方:
    # ROI 1〜14 をまとめて処理
    uv run python src/tools/kinetic_plot.py scripts/limit

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

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

# ----------------------------------------------------------
# ROI → ラベル
# ----------------------------------------------------------
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


def _safe_path_arg(arg: str) -> str:
    """先頭に誤って '-' を付けたパスを救済する。"""
    if arg.startswith("-") and os.path.exists(arg[1:]):
        return arg[1:]
    return arg


def _read_param_file(path: str):
    """1つのパラメータファイル (ka.txt など) を float で読む。"""
    if not os.path.exists(path):
        return None
    try:
        return float(np.genfromtxt(path))
    except Exception:
        return None


def _read_corr(seed_dir: str):
    """seed ディレクトリ内の correlation.txt を読む。"""
    path = os.path.join(seed_dir, "correlation.txt")
    if not os.path.exists(path):
        return None
    try:
        return float(np.genfromtxt(path))
    except Exception:
        return None


def collect_kinetics_and_best(base_dir: str):
    """
    base_dir (= scripts/limit/roi_X/band_full など) 以下の seed_* を走査して
    Kinetics パラメータとベスト seed を集計する。

    Returns
    -------
    data : dict[str, np.ndarray]
        "ka", "kfi", "kfr", "ksi", "ksr" の各キーに 1次元配列。
    seeds_ok : list[int]
        上記 data と同じ順序の seed 番号。
    best_seed : int | None
        correlation.txt が最大の seed 番号（見つからなければ None）。
    best_corr : float | None
        best_seed の相関係数。
    best_params : dict[str, float] | None
        best_seed に対応するパラメータ（ka,kfi,kfr,ksi,ksr）。
    """
    keys = ["ka", "kfi", "kfr", "ksi", "ksr"]
    vals = {k: [] for k in keys}
    seeds_ok: list[int] = []

    best_seed = None
    best_corr = None
    best_params = None

    seed_dirs = sorted(glob.glob(os.path.join(base_dir, "seed_*")))
    for sd in seed_dirs:
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

        # 相関係数
        corr = _read_corr(sd)
        # 最良 seed 更新
        if corr is not None and np.isfinite(corr):
            if (best_corr is None) or (corr > best_corr):
                best_corr = float(corr)
                best_seed = seed_num
                best_params = dict(pvals)

        # 分布用に追加
        for k in keys:
            vals[k].append(pvals[k])
        seeds_ok.append(seed_num)

    # numpy 配列に変換
    data = {k: np.asarray(v, dtype=float) for k, v in vals.items()}

    return data, seeds_ok, best_seed, best_corr, best_params


def _plot_boxplots(data: dict, out_dir: str, roi_label: str):
    """線形軸・対数軸の箱ひげ図を保存。"""
    keys = ["ka", "kfi", "kfr", "ksi", "ksr"]
    x = [data[k] for k in keys]

    # linear
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(x, showmeans=True)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels(keys)
    ax.set_title(f"Kinetics parameters (linear) - {roi_label}")
    ax.set_ylabel("parameter value")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kinetic_linear.pdf"), bbox_inches="tight")
    plt.close(fig)

    # log10
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    # ゼロ以下は描けないので、最低値を少し持ち上げる
    x_log = []
    for arr in x:
        arr = np.asarray(arr, float)
        # 負値は絶対値をとってから log10 にする
        abs_arr = np.abs(arr)
        eps = np.min(abs_arr[abs_arr > 0]) if np.any(abs_arr > 0) else 1e-6
        arr_safe = np.where(abs_arr > 0, abs_arr, eps)
        x_log.append(np.log10(arr_safe))

    ax.boxplot(x_log, showmeans=True)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels(keys)
    ax.set_title(f"Kinetics parameters (log10 |value|) - {roi_label}")
    ax.set_ylabel("log10 |parameter|")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kinetic_log.pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_scatter_matrix(data: dict, out_dir: str, roi_label: str):
    """単純な散布図行列を作る。"""
    keys = ["ka", "kfi", "kfr", "ksi", "ksr"]
    n = len(keys)

    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # 対角成分はヒストグラム
                arr = np.asarray(data[keys[i]], float)
                ax.hist(arr, bins=15)
            else:
                xi = np.asarray(data[keys[j]], float)
                yi = np.asarray(data[keys[i]], float)
                ax.scatter(xi, yi, s=10, alpha=0.5)
            if i == n - 1:
                ax.set_xlabel(keys[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(keys[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Kinetics parameter scatter matrix - {roi_label}", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "kinetic_scatter_matrix.pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_stats(data: dict, out_path: str, seeds_ok: list[int], best_seed, best_corr, best_params):
    keys = ["ka", "kfi", "kfr", "ksi", "ksr"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Kinetics parameter summary stats\n")
        f.write(f"n_seeds = {len(seeds_ok)}\n")
        f.write(f"seeds   = {seeds_ok}\n\n")

        for k in keys:
            arr = np.asarray(data[k], float)
            if arr.size == 0:
                continue
            f.write(f"[{k}]\n")
            f.write(f"  n      = {arr.size}\n")
            f.write(f"  mean   = {np.mean(arr):.6f}\n")
            f.write(f"  median = {np.median(arr):.6f}\n")
            f.write(f"  std    = {np.std(arr):.6f}\n")
            f.write(f"  min    = {np.min(arr):.6f}\n")
            f.write(f"  max    = {np.max(arr):.6f}\n\n")

        f.write("\n[best_seed_by_correlation]\n")
        f.write(f"  best_seed = {best_seed}\n")
        f.write(f"  best_corr = {best_corr}\n")
        if best_params is not None:
            for k in keys:
                f.write(f"  {k} = {best_params.get(k, float('nan')):.6f}\n")


def _plot_best_seed_bar(best_params: dict[str, float], out_dir: str, roi_label: str, best_seed, best_corr):
    keys = ["ka", "kfi", "kfr", "ksi", "ksr"]
    vals = [best_params[k] for k in keys]

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(len(keys)), vals)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys)
    title = f"Best seed parameters (seed={best_seed:02d}, corr={best_corr:.3f})\n{roi_label}"
    ax.set_title(title)
    ax.set_ylabel("parameter value")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kinetic_best_seed_bar.pdf"), bbox_inches="tight")
    plt.close(fig)


def run_for_one_roi(root_dir: str, roi: int, objective: str):
    """
    1つの ROI について Kinetics パラメータ分布と
    ベスト seed のパラメータを可視化する。
    """
    base_dir = os.path.join(root_dir, f"roi_{roi}", objective)
    if not os.path.isdir(base_dir):
        print(f"[WARN] base_dir が存在しません: {base_dir}")
        return

    roi_label = ROI_LABELS.get(roi, f"ROI {roi}")

    out_dir = os.path.join(base_dir, "kinetic_plot")
    os.makedirs(out_dir, exist_ok=True)

    data, seeds_ok, best_seed, best_corr, best_params = collect_kinetics_and_best(base_dir)
    if not seeds_ok:
        print(f"[WARN] ROI {roi}: 有効な seed がありませんでした。")
        return

    # 箱ひげ図・散布図行列
    _plot_boxplots(data, out_dir, roi_label)
    _plot_scatter_matrix(data, out_dir, roi_label)

    # 統計 & ベスト seed 情報
    stats_path = os.path.join(out_dir, "kinetic_stats.txt")
    _save_stats(data, stats_path, seeds_ok, best_seed, best_corr, best_params)

    # ベスト seed の棒グラフ
    if best_seed is not None and best_params is not None and best_corr is not None:
        _plot_best_seed_bar(best_params, out_dir, roi_label, best_seed, best_corr)

    print(f"=== kinetic_plot DONE for ROI {roi} ===")
    print(f"  base_dir : {base_dir}")
    print(f"  out_dir  : {out_dir}")
    print(f"  seeds    : {seeds_ok}")
    print(f"  best_seed: {best_seed} (corr={best_corr})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir",
        help="例: scripts/limit （内部で roi_1〜roi_14/<objective>/ を検索）",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="band_full",
        help="目的関数ディレクトリ名 (default: band_full)",
    )
    parser.add_argument(
        "--roi",
        type=int,
        help="特定の ROI のみ処理したい場合に指定 (1〜14)。未指定なら全 ROI を処理。",
    )
    args = parser.parse_args()

    root_dir = _safe_path_arg(args.root_dir)
    objective = args.objective

    if not os.path.isdir(root_dir):
        print(f"ERROR: root_dir が存在しません: {root_dir}")
        sys.exit(1)

    if args.roi is not None:
        roi_list = [int(args.roi)]
    else:
        roi_list = list(range(1, 15))

    for roi in roi_list:
        print("----------------------------------------")
        print(f"[RUN] ROI {roi}")
        run_for_one_roi(root_dir, roi, objective)


if __name__ == "__main__":
    main()
