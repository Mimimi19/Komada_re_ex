# -*- coding: utf-8 -*-
"""
src/tools/block_corr_boxplot.py

ROI 平均応答と LNK モデル予測 (A_state) から、
刺激をブロックごとに分割して Spearman 相関係数を計算し、
ブロック間・ROI 間で箱ひげ図を比較するツール。

想定ディレクトリ構造:
    data/ret2p/roi_ave/response_ave_roi1.txt  (教師データ: ROI 平均)
    data/ret2p/roi_ave/response_ave_roi2.txt
    ...

    <root_dir>/roi_1/band_full/seed_01/state/A_state.txt
    <root_dir>/roi_1/band_full/seed_02/state/A_state.txt
    ...
    <root_dir>/roi_14/band_full/seed_30/state/A_state.txt

刺激 (chirp_stim_64Hz_bilinear.txt) は 64Hz サンプリング (dt=1/64=0.015625s) で
全長 32 秒 (2048 サンプル) と仮定する。

ブロックの定義（秒単位）:
    Block1:  0 〜 10 s   （ステップ＋プレリード）
    Block2: 10 〜 20 s   （F-chirp 領域 ≒ 10秒）
    Block3: 20 〜 30 s   （A-chirp 領域 ≒ 10秒）
    Block23: 10 〜 30 s  （F-chirp + A-chirp をまとめた区間）
    All:     0 〜 32 s   （全区間）

各 seed について上記各ブロックの Spearman 相関係数を計算し、
ROI ごとに箱ひげ図を出力する。

出力 (従来機能):
    <root_dir>/block_corr/roi_<ROI>/
        roi<ROI>_block_corr_boxplot.pdf      # 各ブロック(ALL含む)を並べた箱ひげ図
        roi<ROI>_block_corr_stats.txt        # ブロックごとの統計情報

追加出力 (今回追加した機能):
    <root_dir>/block_corr/
        block_Block1_step_all_roi_boxplot.pdf
        block_Block2_F_all_roi_boxplot.pdf
        block_Block3_A_all_roi_boxplot.pdf
        block_Block23_F+A_all_roi_boxplot.pdf

    ※ ALL ブロックは ROI 間比較図は作らない（要望通り ALL 以外のみ）

使い方:
    uv run python src/tools/block_corr_boxplot.py scripts/spring04/scripts/limit 

オプション:
    --objective  : band_full 以外の目的関数ディレクトリを使いたい場合
    --resp-root  : ROI 平均応答ファイルが置いてあるディレクトリ
                   (default: data/ret2p/roi_ave)
    --roi-min / --roi-max : ROI の範囲 (default: 1〜14)
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント（あれば）
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


# ==========================================================
# ROI → ラベル（グラフタイトルなどに使う）
# ==========================================================
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


# ==========================================================
# ブロック定義（秒単位）
# ==========================================================
BLOCK_DEFS = {
    "Block1_step":   (0.0, 10.0),   # 初期ステップ＋プレリード
    "Block2_F":      (10.0, 20.0),  # F-chirp 領域（約 10 秒）
    "Block3_A":      (20.0, 30.0),  # A-chirp 領域（約 10 秒）
    "Block2+3_F+A":   (10.0, 30.0),  # F + A
    "All":           (0.0, 32.0),   # 刺激全体
}


# ==========================================================
# ユーティリティ
# ==========================================================
def _safe_path_arg(arg: str) -> str:
    """先頭に誤って '-' を付けたパスを救済する。"""
    if arg.startswith("-") and os.path.exists(arg[1:]):
        return arg[1:]
    return arg


def _load_1d(path: str) -> np.ndarray:
    """1次元テキストを読み込み（float）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.genfromtxt(path, dtype=float)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """
    SciPy なしで rankdata 相当を実装（同値は平均順位）。
    """
    a = np.asarray(a)
    n = a.size
    sorter = np.argsort(a)
    inv = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i + 1
        # 同じ値が続く区間 [i, j)
        while j < n and a[sorter[j]] == a[sorter[i]]:
            j += 1
        # 平均順位（1-origin）
        rank = 0.5 * (i + j - 1) + 1.0
        inv[sorter[i:j]] = rank
        i = j
    return inv


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman の順位相関係数を計算（SciPy なし版）。
    x, y は同じ長さの 1次元配列を想定。
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(len(x), len(y))
    if n < 3:
        return np.nan
    x = x[:n]
    y = y[:n]

    rx = _rankdata(x)
    ry = _rankdata(y)

    # 相関係数（Pearson）を rank に対して計算
    vx = np.var(rx)
    vy = np.var(ry)
    if vx <= 1e-12 or vy <= 1e-12:
        return np.nan
    cov = np.mean((rx - rx.mean()) * (ry - ry.mean()))
    return float(cov / np.sqrt(vx * vy))


def _get_dt_from_any_seed(roi_dir: str, objective: str, default_dt: float = 0.015625) -> float:
    """
    hydra の config.yaml があれば data.dt を拾う…が、
    ここでは簡単のため「見つからなければ default_dt=0.015625」を使う。
    """
    return default_dt


def _collect_seed_dirs(base_dir: str) -> list[str]:
    """base_dir/seed_*/ を列挙。"""
    return [d for d in sorted(glob.glob(os.path.join(base_dir, "seed_*"))) if os.path.isdir(d)]


# ==========================================================
# コア処理: 1 ROI についてブロック別相関を集計 & 箱ひげ図出力
#   戻り値として block_corr を返すように変更
# ==========================================================
def process_one_roi(root_dir: str, roi: int, objective: str, resp_root: str):
    """
    1 つの ROI について、
    - ROI 平均応答 (教師データ)
    - 各 seed の予測 (A_state)
    を読み込み、ブロックごとの Spearman 相関を計算して箱ひげ図を出力。

    戻り値:
        block_corr : dict[str, list[float]]  （ブロック名 -> seed ごとの相関リスト）
        何も計算できなかった場合は None
    """
    roi_label = ROI_LABELS.get(roi, f"ROI {roi}")
    roi_dir = os.path.join(root_dir, f"roi_{roi}", objective)
    if not os.path.isdir(roi_dir):
        print(f"[WARN] ROI {roi}: ディレクトリが存在しません: {roi_dir}")
        return None

    # 教師データ（ROI 平均）
    resp_path = os.path.join(resp_root, f"response_ave_roi{roi}.txt")
    try:
        resp = _load_1d(resp_path)
    except FileNotFoundError:
        print(f"[WARN] ROI {roi}: ROI 平均応答ファイルがありません: {resp_path}")
        return None

    # dt とサンプル数
    dt = _get_dt_from_any_seed(roi_dir, objective)
    n_resp = len(resp)

    # ブロックのインデックス範囲（サンプル番号）を事前に計算
    block_indices = {}
    for bname, (t0, t1) in BLOCK_DEFS.items():
        i0 = int(round(t0 / dt))
        i1 = int(round(t1 / dt))
        i0 = max(0, min(i0, n_resp))
        i1 = max(0, min(i1, n_resp))
        if i1 <= i0:
            print(f"[WARN] ROI {roi}: block {bname} の範囲が空です (t0={t0}, t1={t1})")
        block_indices[bname] = (i0, i1)

    # seed ディレクトリ列挙
    base_dir = roi_dir   # = root_dir/roi_X/objective
    seed_dirs = _collect_seed_dirs(base_dir)
    if not seed_dirs:
        print(f"[WARN] ROI {roi}: seed_* がありません: {base_dir}")
        return None

    # ブロックごとの相関を保存する dict: bname -> list[float]
    block_corr = {bname: [] for bname in BLOCK_DEFS.keys()}

    # 各 seed についてブロック別相関を計算
    for sd in seed_dirs:
        seed_name = os.path.basename(sd)
        state_path = os.path.join(sd, "state", "A_state.txt")
        if not os.path.exists(state_path):
            print(f"[INFO] ROI {roi} {seed_name}: A_state.txt が無いのでスキップ")
            continue

        try:
            pred = _load_1d(state_path)
        except FileNotFoundError:
            print(f"[INFO] ROI {roi} {seed_name}: A_state.txt 読み込み失敗でスキップ")
            continue

        n = min(n_resp, len(pred))
        if n < 10:
            print(f"[INFO] ROI {roi} {seed_name}: データ長が短すぎるのでスキップ (n={n})")
            continue

        resp_c = resp[:n]
        pred_c = pred[:n]

        for bname, (i0, i1) in block_indices.items():
            if i1 <= i0 or i1 > n:
                continue
            r_seg = resp_c[i0:i1]
            p_seg = pred_c[i0:i1]
            if len(r_seg) < 10:
                continue

            rho = _spearmanr(r_seg, p_seg)
            if np.isfinite(rho):
                block_corr[bname].append(rho)

    # 1つも有効な相関がない場合は終了
    any_valid = any(len(v) > 0 for v in block_corr.values())
    if not any_valid:
        print(f"[WARN] ROI {roi}: ブロック相関が一つも計算できませんでした。")
        return None

    # 出力ディレクトリ（従来どおり ROI ごと）
    out_dir = os.path.join(root_dir, "block_corr", f"roi_{roi}")
    os.makedirs(out_dir, exist_ok=True)

    # 統計情報をテキスト出力
    stats_path = os.path.join(out_dir, f"roi{roi}_block_corr_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"ROI {roi}  ({roi_label})  objective={objective}\n\n")
        for bname, vals in block_corr.items():
            arr = np.asarray(vals, float)
            if arr.size == 0:
                f.write(f"[{bname}]\n  n=0\n\n")
                continue
            f.write(f"[{bname}]\n")
            f.write(f"  n      = {arr.size}\n")
            f.write(f"  mean   = {np.mean(arr):.6f}\n")
            f.write(f"  median = {np.median(arr):.6f}\n")
            f.write(f"  std    = {np.std(arr):.6f}\n")
            f.write(f"  min    = {np.min(arr):.6f}\n")
            f.write(f"  max    = {np.max(arr):.6f}\n\n")

    # ROI 内でのブロック比較の箱ひげ図（従来どおり）
    order = ["All", "Block1_step", "Block2_F", "Block3_A", "Block2+3_F+A"]
    values = [block_corr[b] for b in order if len(block_corr[b]) > 0]
    labels = [b for b in order if len(block_corr[b]) > 0]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(values, labels=labels, showmeans=True, vert=False)

    ax.set_title(f"{roi_label} のブロック別 Spearman 相関分布", fontsize=14)
    ax.set_xlabel("spearman の順位相関係数")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    pdf_path = os.path.join(out_dir, f"roi{roi}_block_corr_boxplot.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"=== block_corr_boxplot DONE (ROI {roi}) ===")
    print(f"  roi_dir    : {roi_dir}")
    print(f"  resp_file  : {resp_path}")
    print(f"  seed_dirs  : {len(seed_dirs)} 個")
    print(f"  out_dir    : {out_dir}")
    print(f"  stats_path : {stats_path}")
    print(f"  pdf_path   : {pdf_path}")

    # ここで per-ROI のブロック相関を返す
    return block_corr


# ==========================================================
# 追加: ブロックごとに ROI を並べた箱ひげ図
# ==========================================================
def plot_block_across_rois(block_name: str, data_dict: dict[int, list[float]], root_dir: str):
    """
    1つのブロック (block_name) について、
    ROI ごとの相関値分布 (seed 分布) を 1 枚の図にまとめる。

    data_dict: roi -> list[float]
    出力先: <root_dir>/block_corr/block_<block_name>_all_roi_boxplot.pdf
    """
    # 相関値が 1 つもない ROI は除外
    rois = [roi for roi, vals in data_dict.items() if len(vals) > 0]
    if len(rois) == 0:
        print(f"[WARN] Block {block_name}: 有効な ROI がありませんでした。")
        return

    rois = sorted(rois)
    values = [data_dict[roi] for roi in rois]
    labels = [ROI_LABELS.get(roi, f"ROI {roi}") for roi in rois]

    out_dir = os.path.join(root_dir, "block_corr")
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(values, labels=labels, showmeans=True, vert=False)

    ax.set_title(f"{block_name}の ROI 間 Spearman 相関比較")
    ax.set_xlabel("spearman の順位相関係数")
    ax.grid(True, linestyle="--", alpha=0.3)
    # ROI ラベルが多いので少し小さめ & 回転
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    safe_block = block_name.replace("/", "_").replace(" ", "_")
    pdf_path = os.path.join(out_dir, f"block_{safe_block}_all_roi_boxplot.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"=== block_corr across ROIs DONE (block={block_name}) ===")
    print(f"  pdf_path   : {pdf_path}")


# ==========================================================
# メイン
# ==========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root_dir",
        help="例: scripts/limit （内部で roi_1〜roi_14/<objective>/ を探索）",
    )
    ap.add_argument(
        "--objective",
        type=str,
        default="band_full",
        help="目的関数ディレクトリ名 (default: band_full)",
    )
    ap.add_argument(
        "--resp-root",
        type=str,
        default="data/ret2p/roi_ave",
        help="ROI平均応答ファイルのディレクトリ (default: data/ret2p/roi_ave)",
    )
    ap.add_argument(
        "--roi-min",
        type=int,
        default=1,
        help="最初の ROI 番号 (default: 1)",
    )
    ap.add_argument(
        "--roi-max",
        type=int,
        default=14,
        help="最後の ROI 番号 (default: 14)",
    )
    args = ap.parse_args()

    root_dir = _safe_path_arg(args.root_dir)
    objective = args.objective
    resp_root = args.resp_root

    if not os.path.isdir(root_dir):
        print(f"ERROR: root_dir が存在しません: {root_dir}")
        sys.exit(1)

    if not os.path.isdir(resp_root):
        print(f"ERROR: resp-root ディレクトリが存在しません: {resp_root}")
        sys.exit(1)

    # 全 ROI 分の block_corr をまとめて保持:
    #   global_block_corr[block_name][roi] = list[float]
    global_block_corr: dict[str, dict[int, list[float]]] = {
        bname: {} for bname in BLOCK_DEFS.keys()
    }

    for roi in range(args.roi_min, args.roi_max + 1):
        print("\n----------------------------------------")
        print(f"[RUN] ROI {roi}")
        block_corr = process_one_roi(root_dir, roi, objective, resp_root)
        if block_corr is None:
            continue

        # ROI ごとの結果を global に格納
        for bname, vals in block_corr.items():
            if len(vals) == 0:
                continue
            if roi not in global_block_corr[bname]:
                global_block_corr[bname][roi] = []
            global_block_corr[bname][roi].extend(vals)

    # ここから追加機能:
    #  ALL 以外のブロックについて、ROI を並べた比較図を作成
    for bname in BLOCK_DEFS.keys():
        if bname == "All":
            continue  # ALL はスキップ（要望どおり）
        data_dict = global_block_corr.get(bname, {})
        if not data_dict:
            print(f"[INFO] Block {bname}: どの ROI にもデータが無いのでスキップ")
            continue
        plot_block_across_rois(bname, data_dict, root_dir)


if __name__ == "__main__":
    main()
