# data/tools/claster_ave.py
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
from typing import Dict, List, Tuple

"""
    uv run data/tools/claster_ave.py
"""

def load_roi_ids(cluster_id_path: str) -> np.ndarray:
    """
    cluster_id.txt を読み込み、NaN を除いた ROI id（int）配列を返す。
    """
    raw = np.genfromtxt(cluster_id_path, dtype=float)
    if raw.ndim != 1:
        raw = raw.reshape(-1)

    valid = ~np.isnan(raw)
    ids = raw[valid].astype(int)

    # 元の列数（=cell数）も必要なので、NaNは -1 として保持する版も返す
    full = np.full_like(raw, fill_value=-1, dtype=int)
    full[valid] = ids
    return full


def build_index_map(roi_full: np.ndarray) -> Dict[int, np.ndarray]:
    """
    roi_id -> indices(np.ndarray) の辞書を作る（-1 は無視）。
    """
    uniq = sorted([int(x) for x in np.unique(roi_full) if x >= 0])
    index_map: Dict[int, np.ndarray] = {}
    for rid in uniq:
        index_map[rid] = np.where(roi_full == rid)[0]
    return index_map


def stream_mean_by_roi(
    response_mat_path: str,
    index_map: Dict[int, np.ndarray],
    n_cells_expected: int,
) -> Tuple[np.ndarray, List[int]]:
    """
    response_data_repeat_1.txt を1行ずつ読み、各ROIごとの平均時系列を返す。
    戻り: (means[T, n_roi], roi_ids_list)
    """
    roi_ids = sorted(index_map.keys())
    n_roi = len(roi_ids)

    means_rows: List[np.ndarray] = []

    with open(response_mat_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # 高速に1行をパース（空白区切り想定）
            row = np.fromstring(line, sep=" ", dtype=np.float64)

            # もしタブ区切り等で空だった場合はフォールバック
            if row.size == 0:
                row = np.fromstring(line.replace("\t", " "), sep=" ", dtype=np.float64)

            if row.size != n_cells_expected:
                raise ValueError(
                    f"Row {line_no}: expected {n_cells_expected} cols, got {row.size}. "
                    f"cluster_id.txt と response_data_repeat_1.txt の列数が一致していません。"
                )

            out = np.empty(n_roi, dtype=np.float64)
            for j, rid in enumerate(roi_ids):
                idx = index_map[rid]
                # 応答に NaN が混じっても平均できるように nanmean
                out[j] = np.nanmean(row[idx]) if idx.size > 0 else np.nan
            means_rows.append(out)

    means = np.vstack(means_rows)  # shape=(T, n_roi)
    return means, roi_ids


def save_means(means: np.ndarray, roi_ids: List[int], out_dir: str) -> None:
    """
    data/ret2p/response_ave_roi{ID}.txt として保存（1列: 時系列）。
    """
    os.makedirs(out_dir, exist_ok=True)
    T = means.shape[0]
    for j, rid in enumerate(roi_ids):
        out_path = os.path.join(out_dir, f"response_ave_roi{rid}.txt")
        # 1列で保存
        np.savetxt(out_path, means[:, j].reshape(T, 1), fmt="%.6f")
        print(f"[saved] {out_path} (T={T})")


def main():
    # ========= 入力（必要に応じてここだけ編集） =========
    CLUSTER_ID_PATH = "data/ret2p/cluster_idx.txt"
    RESPONSE_MAT_PATH = "data/ret2p/trim/11101x1/response_data_repeat_1.txt"
    OUT_DIR = "data/ret2p/roi_ave/roi_ave_Block1-2"
    # ================================================

    roi_full = load_roi_ids(CLUSTER_ID_PATH)
    n_cells = roi_full.size
    print("=== claster_ave.py ===")
    print(f"cluster_id : {CLUSTER_ID_PATH} (n_cells={n_cells})")
    print(f"response   : {RESPONSE_MAT_PATH}")
    print(f"out_dir    : {OUT_DIR}")

    index_map = build_index_map(roi_full)
    roi_ids = sorted(index_map.keys())
    print(f"roi types  : {roi_ids}")
    for rid in roi_ids:
        print(f"  roi {rid}: n={index_map[rid].size}")

    means, roi_ids = stream_mean_by_roi(RESPONSE_MAT_PATH, index_map, n_cells_expected=n_cells)
    print(f"computed means: T={means.shape[0]}, n_roi={means.shape[1]}")

    save_means(means, roi_ids, OUT_DIR)
    print("done.")


if __name__ == "__main__":
    main()
