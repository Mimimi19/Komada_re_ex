# src/tools/limit_ave_plot.py
# -*- coding: utf-8 -*-
"""
各 ROI について、教師データ（平均応答）と
各 seed の A_state(t)（=Kinetics ブロックの活性状態）を比較するプロットを作るツール。

(1) 全 seed overlay + 平均 + 教師データ
(2) 相関係数が最大の「ベスト seed」だけを重ねたプロット

想定ディレクトリ構造:
    <root_dir>/
      roi_1/
        band_full/
          seed_01/
            state/A_state.txt
            correlation.txt
          ...
          seed_30/
            ...
      ...
      roi_14/
        band_full/
          ...

教師データ (ROI 平均応答):
    data/ret2p/roi_ave/response_ave_roi{roi}.txt

出力:
    <root_dir>/roi_<ROI>/<objective>/ave_plot/
      limit_ave_plot.pdf             ... 全 seed overlay + 平均 + 教師
      limit_ave_plot_best_seed.pdf   ... ベスト seed のみ + 教師
      limit_ave_best_seed.txt        ... ベスト seed と相関係数

使い方:
    # ROI 1〜14 をまとめて処理
    uv run python src/tools/limit_ave_plot.py scripts/spring04/scripts/limit 

    # 特定 ROI だけ処理
    uv run python src/tools/limit_ave_plot.py scripts/spring04/scripts/limit --roi 11

    # 目的関数ディレクトリを変える場合
    uv run python src/tools/limit_ave_plot.py scripts/limit --objective band_low_only
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


def _safe_path_arg(arg: str) -> str:
    if arg.startswith("-") and os.path.exists(arg[1:]):
        return arg[1:]
    return arg


def _load_1d_txt(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.genfromtxt(path).astype(float)


def _normalize_zero_mean_max1(x: np.ndarray) -> np.ndarray:
    """平均0・最大絶対値1 に正規化。"""
    x = np.asarray(x, float)
    x = x - np.mean(x)
    m = np.max(np.abs(x))
    if m > 1e-12:
        x = x / m
    return x


def _read_corr(seed_dir: str):
    path = os.path.join(seed_dir, "correlation.txt")
    if not os.path.exists(path):
        return None
    try:
        return float(np.genfromtxt(path))
    except Exception:
        return None


def collect_astates_and_best(base_dir: str):
    """
    base_dir (= scripts/limit/roi_X/band_full など) 以下の seed_* を走査して
    A_state(t) とベスト seed を集計する。

    Returns
    -------
    traces : list[np.ndarray]
        各 seed の A_state 生データ（まだ正規化・長さ調整前）
    seeds  : list[int]
        traces と同じ順序の seed 番号
    best_seed : int | None
        correlation.txt が最大の seed 番号（見つからなければ None）
    best_corr : float | None
        best_seed の相関係数
    """
    traces = []
    seeds = []
    best_seed = None
    best_corr = None

    seed_dirs = sorted(glob.glob(os.path.join(base_dir, "seed_*")))
    for sd in seed_dirs:
        base = os.path.basename(sd)
        try:
            seed_num = int(base.split("_")[1])
        except Exception:
            continue

        a_path = os.path.join(sd, "state", "A_state.txt")
        if not os.path.exists(a_path):
            continue
        try:
            trace = np.genfromtxt(a_path).astype(float)
        except Exception:
            continue

        traces.append(trace)
        seeds.append(seed_num)

        # 相関係数
        corr = _read_corr(sd)
        if corr is not None and np.isfinite(corr):
            if (best_corr is None) or (corr > best_corr):
                best_corr = float(corr)
                best_seed = seed_num

    return traces, seeds, best_seed, best_corr


def _plot_all_seeds(resp_n, preds_n, t, out_path: str, roi_label: str):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)

    # 全 seed の A_state
    for y in preds_n:
        ax.plot(t, y, color="tab:green", alpha=0.25, linewidth=0.7)

    # 予測平均
    if preds_n:
        mean_pred = np.mean(np.stack(preds_n, axis=0), axis=0)
        ax.plot(t, mean_pred, color="tab:red", linewidth=1.8, label="Mean A_state")

    # 教師データ
    ax.plot(t, resp_n, color="tab:blue", linewidth=1.5, label="BC response")

    # ax.set_title(f"LNKモデルの予測応答とROIごとの平均応答 (全 seed) - {roi_label}", fontsize=18)
    ax.set_title(f" LNK Model Predictions and Subtype Averaged Responses - {roi_label}", fontsize=18)    
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Normalized response", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_best_seed(resp_n, best_pred, t, out_path: str, roi_label: str, best_seed: int | None, best_corr: float | None):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)

    # ベスト seed の A_state
    ax.plot(t, best_pred, color="tab:green", linewidth=1.8,
            label=f"A_state (Best seed )")

    # 教師データ
    ax.plot(t, resp_n, color="tab:blue", linewidth=1.5, label="BC response")

    title_extra = ""
    if best_seed is not None:
        title_extra += f"seed={best_seed:02d}"
    if best_corr is not None:
        if title_extra:
            title_extra += ", "
        title_extra += f"corr={best_corr:.3f}"

    if title_extra:
        title = f"LNK A_state (best seed) vs BC Responses - {roi_label}\n({title_extra})"
    else:
        title = f"LNK A_state (best seed) vs BC Responses - {roi_label}"

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Normalized response", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_for_one_roi(root_dir: str, roi: int, objective: str, dt: float = 0.015625):
    base_dir = os.path.join(root_dir, f"roi_{roi}", objective)
    if not os.path.isdir(base_dir):
        print(f"[WARN] base_dir が存在しません: {base_dir}")
        return

    # 教師データ
    resp_path = os.path.join("data", "ret2p", "roi_ave", f"response_ave_roi{roi}.txt")
    try:
        resp = _load_1d_txt(resp_path)
    except FileNotFoundError:
        print(f"[WARN] 教師データが見つかりません: {resp_path}")
        return

    traces, seeds, best_seed, best_corr = collect_astates_and_best(base_dir)
    if not traces:
        print(f"[WARN] ROI {roi}: A_state が 1 本も見つかりませんでした。")
        return

    # 長さ揃え & 正規化
    min_len = min(len(resp), *(len(tr) for tr in traces))
    resp = resp[:min_len]
    resp_n = _normalize_zero_mean_max1(resp)

    preds_n = []
    best_pred_n = None

    for tr, s in zip(traces, seeds):
        tr = tr[:min_len]
        tr_n = _normalize_zero_mean_max1(tr)
        preds_n.append(tr_n)
        if best_seed is not None and s == best_seed:
            best_pred_n = tr_n

    # best_seed が見つからなかった場合は 1本目を採用
    if best_pred_n is None:
        best_pred_n = preds_n[0]
        if best_seed is None and seeds:
            best_seed = seeds[0]

    t = np.arange(min_len) * dt
    roi_label = ROI_LABELS.get(roi, f"ROI {roi}")

    out_dir = os.path.join(base_dir, "ave_plot")
    os.makedirs(out_dir, exist_ok=True)

    # (1) 全 seed overlay → PDF
    out_pdf_all = os.path.join(out_dir, "limit_ave_plot.pdf")
    _plot_all_seeds(resp_n, preds_n, t, out_pdf_all, roi_label)

    # (2) ベスト seed → PDF
    out_pdf_best = os.path.join(out_dir, "limit_ave_plot_best_seed.pdf")
    _plot_best_seed(resp_n, best_pred_n, t, out_pdf_best, roi_label, best_seed, best_corr)

    # ベスト seed 情報をテキスト出力
    best_info_path = os.path.join(out_dir, "limit_ave_best_seed.txt")
    with open(best_info_path, "w", encoding="utf-8") as f:
        f.write("# best seed by correlation\n")
        f.write(f"best_seed = {best_seed}\n")
        f.write(f"best_corr = {best_corr}\n")
        f.write(f"seeds     = {seeds}\n")

    print(f"=== limit_ave_plot DONE for ROI {roi} ===")
    print(f"  base_dir  : {base_dir}")
    print(f"  out_dir   : {out_dir}")
    print(f"  seeds     : {seeds}")
    print(f"  best_seed : {best_seed} (corr={best_corr})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root_dir",
        help="例: scripts/limit （内部で roi_1〜roi_14/<objective>/seed_*/ を検索）",
    )
    ap.add_argument(
        "--objective",
        type=str,
        default="band_full",
        help="目的関数ディレクトリ名 (default: band_full)",
    )
    ap.add_argument(
        "--roi",
        type=int,
        help="特定 ROI のみ処理したい場合に指定 (1〜14)。未指定なら全 ROI を処理。",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=0.015625,
        help="時間刻み (s)。デフォルト 1/64 ≒ 0.015625。",
    )
    args = ap.parse_args()

    root_dir = _safe_path_arg(args.root_dir)
    objective = args.objective
    dt = float(args.dt)

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
        run_for_one_roi(root_dir, roi, objective, dt=dt)


if __name__ == "__main__":
    main()
