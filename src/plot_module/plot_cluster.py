# src/plot_module/plot_cluster.py
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def normalize_for_plot(x: np.ndarray) -> np.ndarray:
    """
    プロット専用正規化：
    - 平均を 0
    - 最大絶対値を 1
    """
    x = x - np.mean(x)
    max_abs = np.max(np.abs(x))
    if max_abs > 1e-12:
        x = x / max_abs
    return x


def load_1d(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64).reshape(-1)


def find_predict_files(cluster_dir: str):
    patterns = [
        os.path.join(cluster_dir, "roi_*", "validation", "predict.txt"),
        os.path.join(cluster_dir, "roi_*", "caiidation", "predict.txt"),
        os.path.join(cluster_dir, "roi_*", "**", "predict.txt"),  # 保険
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    return sorted(list(dict.fromkeys(files)))


def main(
    cluster_dir: str,
    roi_ave_file: str = "data/ret2p/roi_ave/response_ave_roi7.txt",
    dt: float = 0.015625,  # 64Hz想定（必要なら変更）
    out_name: str = "cluster_predicts_vs_roiave.png",
):
    if not os.path.isdir(cluster_dir):
        raise FileNotFoundError(f"cluster_dir not found: {cluster_dir}")

    predict_files = find_predict_files(cluster_dir)
    if len(predict_files) == 0:
        raise FileNotFoundError(f"predict.txt not found under: {cluster_dir}")

    # 実測（グレー）
    y_true = load_1d(roi_ave_file)

    # 予測（複数ROI）
    preds = []
    for p in predict_files:
        try:
            y = load_1d(p)
            if len(y) > 5:
                preds.append(y)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    if len(preds) == 0:
        raise RuntimeError("no valid predict series loaded.")

    # 長さを揃える（最短に合わせる）
    min_len = min([len(y_true)] + [len(p) for p in preds])
    y_true = y_true[:min_len]
    P = np.stack([p[:min_len] for p in preds], axis=0)  # (n_roi, T)
    p_mean = P.mean(axis=0)

    t = np.arange(min_len) * dt

    # 正規化
    y_true_n = normalize_for_plot(y_true)
    P_n = np.stack([normalize_for_plot(p) for p in P], axis=0)
    p_mean_n = normalize_for_plot(p_mean)

    # Plot
    plt.figure(figsize=(20, 6))

    # 実測（グレー）
    plt.plot(
        t, y_true_n,
        color="gray",
        alpha=1.0,
        linewidth=2.5,
        label="ROI7 average (response)"
    )

    # 各ROI予測（薄赤）
    for i in range(P_n.shape[0]):
        plt.plot(
            t, P_n[i],
            color="red",
            alpha=0.5,
            linewidth=1.0
        )

    # 予測平均（濃赤）
    plt.plot(
        t, p_mean_n,
        color="red",
        alpha=1.0,
        linewidth=3.0,
        label=f"Predict mean (n={P_n.shape[0]})"
    )


    cluster_name = os.path.basename(os.path.normpath(cluster_dir))
    plt.title(f"{cluster_name}: ROI predictions + mean vs ROI7 average response", fontsize=14)
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Normalized response", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    out_path = os.path.join(cluster_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("=== Saved ===")
    print(out_path)
    print(f"n_roi={P.shape[0]}  len={min_len}  dt={dt}")


if __name__ == "__main__":
    # ==========================================================
    # ▼▼▼ ここを編集すれば切り替えできます ▼▼▼
    # ==========================================================
    CLUSTER_DIR = "scripts/limit/seed_10"
    ROI_AVE_FILE = "data/ret2p/roi_ave/response_ave_roi9.txt"
    DT = 0.015625
    OUT_NAME = "cluster_9.png"
    # ==========================================================

    main(CLUSTER_DIR, roi_ave_file=ROI_AVE_FILE, dt=DT, out_name=OUT_NAME)
