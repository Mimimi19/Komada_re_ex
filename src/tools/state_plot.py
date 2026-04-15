# =============================================
# src/tools/state_plot.py (ROI 1–14 全対応版)
# =============================================
# -*- coding: utf-8 -*-

"""
状態変数 A, R, I1, I2 の占有率を可視化するツール：
    (1) best seed の占有状態推移
    (2) 全 seed overlay
    (3) 全 seed mean ± std 帯

入力:
    <root>/roi_<ROI>/<objective>/seed_xx/state/{A,R,I1,I2}_state.txt

出力:
    <root>/<objective>/state_plot/roi_<ROI>/
        roi_<ROI>_all_seeds.pdf
        roi_<ROI>_best_seedXX.pdf
        roi_<ROI>_mean_std.pdf

実行例:
    # ROI 1〜14 を全部処理（推奨）
    uv run python src/tools/state_plot.py scripts/limit

    # 特定 ROI だけ処理したい場合
    uv run python src/tools/state_plot.py scripts/limit --roi 11
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


# -----------------------------
# ユーティリティ
# -----------------------------
def _safe_path_arg(arg: str) -> str:
    """先頭に誤って '-' を付けたパスを救済する。"""
    if arg.startswith("-") and os.path.exists(arg[1:]):
        return arg[1:]
    return arg


def _load_1d(path: str):
    """1次元 float テキストを読む。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.genfromtxt(path, dtype=float)


def _try_load_hydra_cfg(seed_dir: str):
    """seed_xx/.hydra/config.yaml を読む。"""
    cfg_path = os.path.join(seed_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _get_dt(seed_dir: str, default: float = 0.015625) -> float:
    """Hydra の data.dt を優先的に使い、なければ default。"""
    cfg = _try_load_hydra_cfg(seed_dir)
    if isinstance(cfg, dict):
        data = cfg.get("data", {})
        if isinstance(data, dict) and data.get("dt") is not None:
            try:
                return float(data["dt"])
            except Exception:
                pass
    return default


def _compute_norm_cum(A, R, I1, I2):
    """
    A,R,I1,I2 から占有率正規化と累積境界線を計算する。
    Returns:
        (cum_A, cum_R, cum_I1, cum_I2)
    """
    n = min(len(A), len(R), len(I1), len(I2))
    A, R, I1, I2 = map(lambda x: np.asarray(x[:n], float), (A, R, I1, I2))
    M = np.vstack([A, R, I1, I2])  # (4, n)
    S = np.sum(M, axis=0)          # (n,)
    mask = S > 1e-12
    M[:, mask] = M[:, mask] / S[mask]
    A, R, I1, I2 = M
    return (A, A + R, A + R + I1, A + R + I1 + I2)


def _read_corr(seed_dir: str):
    """seed_xx/correlation.txt を読み込む。"""
    p = os.path.join(seed_dir, "correlation.txt")
    if not os.path.exists(p):
        return None
    try:
        return float(np.genfromtxt(p))
    except Exception:
        return None


# -----------------------------
# プロット関数群
# -----------------------------
def plot_all_overlay(cum_list, dt, out_path, roi, objective):
    """全 seed の累積境界線を alpha で重ね描き。"""
    if len(cum_list) == 0:
        print(f"[WARN] ROI {roi}: cum_list が空のため overlay はスキップ")
        return

    min_len = min(len(c[0]) for c in cum_list)
    cum_list = [
        (c[0][:min_len], c[1][:min_len], c[2][:min_len], c[3][:min_len])
        for c in cum_list
    ]
    t = np.arange(min_len) * dt
    N = len(cum_list)
    alpha = 1.0 / (N * 0.5)

    fig, ax = plt.subplots(figsize=(10, 5))
    for cA, cR, cI1, cI2 in cum_list:
        ax.plot(t, cA,  color="tab:blue",   alpha=alpha)
        ax.plot(t, cR,  color="tab:orange", alpha=alpha)
        ax.plot(t, cI1, color="tab:green",  alpha=alpha)
        ax.plot(t, cI2, color="tab:red",    alpha=alpha)

    ax.set_title(f"State occupancy overlay (ROI {roi}, {objective})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State occupancy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_best(cum, dt, out_path, roi, seed, objective):
    """ベスト seed の累積境界線のみを描画。"""
    cA, cR, cI1, cI2 = cum
    t = np.arange(len(cA)) * dt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, cA,  color="tab:blue",   label="A")
    ax.plot(t, cR,  color="tab:orange", label="R")
    ax.plot(t, cI1, color="tab:green",  label="I1")
    ax.plot(t, cI2, color="tab:red",    label="I2")

    ax.set_title(f"BEST seed={seed:02d} (ROI {roi}, {objective})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State occupancy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mean_std(cum_list, dt, out_path, roi, objective):
    """全 seed の mean ± std を描画。"""
    if len(cum_list) == 0:
        print(f"[WARN] ROI {roi}: cum_list が空のため mean±std はスキップ")
        return

    min_len = min(len(c[0]) for c in cum_list)
    arr = np.stack(
        [np.stack([c[i][:min_len] for i in range(4)], axis=0) for c in cum_list],
        axis=0,
    )  # (N, 4, T)

    mean = np.mean(arr, axis=0)  # (4, T)
    std = np.std(arr, axis=0)    # (4, T)
    t = np.arange(min_len) * dt

    labels = ["A", "R", "I1", "I2"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(4):
        ax.plot(t, mean[i], color=colors[i], label=f"{labels[i]} mean")
        ax.fill_between(
            t,
            mean[i] - std[i],
            mean[i] + std[i],
            color=colors[i],
            alpha=0.25,
        )

    ax.set_title(f"Mean ± Std (ROI {roi}, {objective})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State occupancy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# ROI 1つ分を処理する関数
# -----------------------------
def process_one_roi(root: str, roi: int, objective: str):
    """
    1つの ROI に対して:
      - best seed を探す
      - 全 seed の cum を集める
      - overlay / best / mean±std を保存

    出力:
        <root>/state_plot/roi_<ROI>/ ...
    """
    roi_in_dir = os.path.join(root, f"roi_{roi}", objective)
    if not os.path.isdir(roi_in_dir):
        print(f"[WARN] ROI {roi}: 入力ディレクトリがありません: {roi_in_dir}")
        return

    # ★ 出力先（新仕様）
    out_dir = os.path.join(root, "state_plot", f"roi_{roi}")
    os.makedirs(out_dir, exist_ok=True)

    best_seed = None
    best_corr = -1e9
    dt = None
    cum_list = []

    for seed in range(1, 31):
        seed_dir = os.path.join(roi_in_dir, f"seed_{seed:02d}")
        state_dir = os.path.join(seed_dir, "state")
        if not os.path.isdir(state_dir):
            continue

        if dt is None:
            dt = _get_dt(seed_dir)

        corr = _read_corr(seed_dir)
        if corr is not None and corr > best_corr:
            best_corr = corr
            best_seed = seed

        try:
            A = _load_1d(os.path.join(state_dir, "A_state.txt"))
            R = _load_1d(os.path.join(state_dir, "R_state.txt"))
            I1 = _load_1d(os.path.join(state_dir, "I1_state.txt"))
            I2 = _load_1d(os.path.join(state_dir, "I2_state.txt"))
        except FileNotFoundError:
            continue

        cum = _compute_norm_cum(A, R, I1, I2)
        cum_list.append(cum)

    if dt is None:
        dt = 0.015625

    if len(cum_list) == 0:
        print(f"[WARN] ROI {roi}: 有効な seed が 1つもありません")
        return

    # (1) overlay
    plot_all_overlay(
        cum_list, dt,
        os.path.join(out_dir, f"roi_{roi}_all_seeds.pdf"),
        roi, objective
    )

    # (2) best seed
    if best_seed is not None:
        best_seed_dir = os.path.join(roi_in_dir, f"seed_{best_seed:02d}", "state")
        A = _load_1d(os.path.join(best_seed_dir, "A_state.txt"))
        R = _load_1d(os.path.join(best_seed_dir, "R_state.txt"))
        I1 = _load_1d(os.path.join(best_seed_dir, "I1_state.txt"))
        I2 = _load_1d(os.path.join(best_seed_dir, "I2_state.txt"))
        cum_best = _compute_norm_cum(A, R, I1, I2)
        plot_best(
            cum_best, dt,
            os.path.join(out_dir, f"roi_{roi}_best_seed{best_seed:02d}.pdf"),
            roi, best_seed, objective
        )

    # (3) mean±std
    plot_mean_std(
        cum_list, dt,
        os.path.join(out_dir, f"roi_{roi}_mean_std.pdf"),
        roi, objective
    )

    print(f"[INFO] ROI {roi}: DONE. best_seed={best_seed}, corr={best_corr:.4f}")
    print(f"       out_dir: {out_dir}")


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", help="例: scripts/limit")
    ap.add_argument("--objective", default="band_full", help="例: band_full / band_low_only")
    ap.add_argument("--roi", type=int, help="特定 ROI のみ処理したい場合に指定 (1〜14)。未指定なら1〜14全部。")
    args = ap.parse_args()

    root = _safe_path_arg(args.root_dir)
    obj = args.objective

    if not os.path.isdir(root):
        print(f"ERROR: root_dir が存在しません: {root}")
        sys.exit(1)

    if args.roi is not None:
        roi_list = [int(args.roi)]
    else:
        roi_list = list(range(1, 15))  # 1〜14

    print(f"root_dir  : {root}")
    print(f"objective : {obj}")
    print(f"ROIs      : {roi_list}")
    print("=========================================")

    for roi in roi_list:
        process_one_roi(root, roi, obj)

    print("=== state_plot ALL DONE ===")


if __name__ == "__main__":
    main()
