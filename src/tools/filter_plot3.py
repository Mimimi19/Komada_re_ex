# src/tools/filter_plot3.py
# -*- coding: utf-8 -*-

"""
非線形関数 u(t) の表示専用ツール。

旧 filter_plot.py では、
    g = _mean0_max1(g)
    u = _mean0_max1(u)
としていたため、非線形出力 u(t) の平均が0にシフトされ、
本来は正の値しか持たない u(t) でも図上では負の値が出ていた。

このツールでは、u(t) について以下2種類のPDFを出力する。

1. raw output:
    N_LNK.main() の純粋な出力をそのまま描画する。
    平均シフトなし、0-1正規化なし。

2. normalized 0-1 output:
    平均シフトは行わず、
    u_norm = (u - min(u)) / (max(u) - min(u))
    により 0〜1 に正規化して描画する。

出力先:
    <root_dir>/roi_<ROI>/<objective>/filter_plot4/

実行例:
    uv run python src/tools/filter_plot3.py scripts/spring04/scripts/limit --objective band_full

ROI14のみ:
    uv run python src/tools/filter_plot3.py scripts/limit --objective band_full --roi 14
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---- path ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)

if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

import components.L_LNK as L_LNK
import components.N_LNK as N_LNK


# ==========================================================
# settings
# ==========================================================

LEGEND_TOP = 5
RNG = np.random.default_rng(0)

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


# ==========================================================
# utility
# ==========================================================

def _safe_path_arg(arg: str) -> str:
    """先頭に誤って '-' を付けたパスを救済する。"""
    if arg.startswith("-") and os.path.exists(arg[1:]):
        return arg[1:]
    return arg


def _load_txt(path: str):
    return np.genfromtxt(path).astype(float)


def _read_param(seed_dir: str, name: str):
    path = os.path.join(seed_dir, f"{name}.txt")
    if not os.path.exists(path):
        return None

    try:
        v = float(np.genfromtxt(path))
        if np.isfinite(v):
            return v
    except Exception:
        pass

    return None


def _read_Ls(seed_dir: str):
    vals = []

    for i in range(1, 200):
        path = os.path.join(seed_dir, f"L{i}.txt")

        if not os.path.exists(path):
            break

        try:
            v = float(np.genfromtxt(path))
            if not np.isfinite(v):
                break
            vals.append(v)
        except Exception:
            break

    if len(vals) == 0:
        return None

    return np.asarray(vals, dtype=float)


def _read_corr(seed_dir: str):
    path = os.path.join(seed_dir, "correlation.txt")

    if not os.path.exists(path):
        return None

    try:
        v = float(np.genfromtxt(path))
        if np.isfinite(v):
            return v
    except Exception:
        pass

    return None


def _var_match_scale_kernel(stim, kernel):
    """
    旧 filter_plot.py と同様、刺激と畳み込み後 g の分散を合わせるために
    kernel をスケールする。
    """
    g = np.convolve(stim, kernel, "same")

    vs = np.var(stim)
    vg = np.var(g)

    if vg <= 1e-12 or vs <= 1e-12:
        return kernel

    return kernel * np.sqrt(vs / vg)


def _normalize_0_1(x):
    """
    平均シフトを行わず、min-max により 0〜1 へ正規化する。

    u_norm = (u - min(u)) / (max(u) - min(u))

    全て同じ値の場合は0配列を返す。
    """
    x = np.asarray(x, dtype=float)

    xmin = np.min(x)
    xmax = np.max(x)

    denom = xmax - xmin

    if denom <= 1e-12:
        return np.zeros_like(x)

    return (x - xmin) / denom


def _subsample_xy(x, y, max_points):
    """
    点数が多すぎる場合、同じ index で x,y を間引く。
    """
    n = min(len(x), len(y))

    x = x[:n]
    y = y[:n]

    if n > max_points:
        idx = RNG.choice(n, max_points, replace=False)
        x = x[idx]
        y = y[idx]

    return x, y


# ==========================================================
# plotting
# ==========================================================

def plot_nonlinear_overlay(seed_data, out_path, title, y_mode, max_points):
    """
    複数seedの非線形関数を重ねて描画する。

    y_mode:
        "raw"        : u_raw を描画
        "normalized" : u_norm_0_1 を描画
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    seed_data_sorted = sorted(seed_data, key=lambda d: d["corr"])
    colors = cm.rainbow(np.linspace(0, 1, len(seed_data_sorted)))

    handles = []

    for i, (d, c) in enumerate(zip(seed_data_sorted, colors)):
        g = d["g"]

        if y_mode == "raw":
            u = d["u_raw"]
        elif y_mode == "normalized":
            u = d["u_norm_0_1"]
        else:
            raise ValueError(f"Unknown y_mode: {y_mode}")

        g_plot, u_plot = _subsample_xy(g, u, max_points)

        label = None
        if i >= len(seed_data_sorted) - LEGEND_TOP:
            label = f"seed {d['seed']} ({d['corr']:.3f})"

        h = ax.scatter(
            g_plot,
            u_plot,
            s=6,
            alpha=0.5,
            color=c,
            label=label,
        )

        if label is not None:
            handles.append(h)

    ax.set_title(title)
    ax.set_xlabel("Linear filter output g(t)")

    if y_mode == "raw":
        ax.set_ylabel("Nonlinearity output u(t)")
    else:
        ax.set_ylabel("Normalized nonlinearity output u(t)")

    ax.grid(True, alpha=0.3)

    if y_mode == "normalized":
        ax.set_ylim(-0.05, 1.05)

    if handles:
        ax.legend(
            handles=handles,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_nonlinear_best(best, out_path, title, y_mode, max_points):
    """
    best seed の非線形関数を描画する。
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    g = best["g"]

    if y_mode == "raw":
        u = best["u_raw"]
    elif y_mode == "normalized":
        u = best["u_norm_0_1"]
    else:
        raise ValueError(f"Unknown y_mode: {y_mode}")

    g_plot, u_plot = _subsample_xy(g, u, max_points)

    ax.scatter(
        g_plot,
        u_plot,
        s=6,
        alpha=0.6,
        color="black",
    )

    ax.set_title(
        title + f"\nseed {best['seed']} corr={best['corr']:.3f}"
    )
    ax.set_xlabel("Linear filter output g(t)")

    if y_mode == "raw":
        ax.set_ylabel("Nonlinearity output u(t)")
    else:
        ax.set_ylabel("Normalized nonlinearity output u(t)")

    ax.grid(True, alpha=0.3)

    if y_mode == "normalized":
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_nonlinear_diagnostics(seed_data, out_path):
    """
    各seedの u_raw の範囲と normalized 後の範囲を保存する。
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# filter_plot4 nonlinear diagnostics\n")
        f.write(
            "seed,corr,"
            "g_min,g_max,"
            "u_raw_min,u_raw_max,u_raw_mean,"
            "u_norm_min,u_norm_max,u_norm_mean\n"
        )

        for d in sorted(seed_data, key=lambda x: x["corr"], reverse=True):
            g = np.asarray(d["g"], dtype=float)
            u_raw = np.asarray(d["u_raw"], dtype=float)
            u_norm = np.asarray(d["u_norm_0_1"], dtype=float)

            f.write(
                f"{d['seed']},{d['corr']:.8f},"
                f"{np.min(g):.8e},{np.max(g):.8e},"
                f"{np.min(u_raw):.8e},{np.max(u_raw):.8e},{np.mean(u_raw):.8e},"
                f"{np.min(u_norm):.8e},{np.max(u_norm):.8e},{np.mean(u_norm):.8e}\n"
            )


# ==========================================================
# core
# ==========================================================

def run_for_roi(base_dir, roi, stim_path, dt, tau, max_points):
    seed_dirs = sorted(glob.glob(os.path.join(base_dir, "seed_*")))

    if not seed_dirs:
        print(f"[WARN] seed directory not found: {base_dir}")
        return

    if not os.path.exists(stim_path):
        print(f"[ERROR] stimulus not found: {stim_path}")
        return

    stim = _load_txt(stim_path)
    stim = stim - np.mean(stim)

    seed_data = []

    for sd in seed_dirs:
        try:
            seed = int(os.path.basename(sd).replace("seed_", ""))
        except Exception:
            continue

        corr = _read_corr(sd)
        if corr is None:
            continue

        alphas = _read_Ls(sd)
        delta = _read_param(sd, "delta")

        a = _read_param(sd, "a")
        kappa = _read_param(sd, "kappa")
        b1 = _read_param(sd, "b1")
        b2 = _read_param(sd, "b2")
        ka = _read_param(sd, "ka")

        # 現在の BaccusModel.py では kappa が保存されていない場合がある。
        # その場合は、実行時と合わせて 1.0 とする。
        if kappa is None:
            kappa = 1.0

        if alphas is None or any(v is None for v in [delta, a, b1, b2, ka]):
            continue

        filter_points = int(tau / dt) + 1

        try:
            kernel, _ = L_LNK.main(
                alphas=alphas,
                delta=delta,
                t=filter_points,
                dt=dt,
                tau=tau,
            )
        except Exception as e:
            print(f"[WARN] kernel reconstruction failed: {sd}: {e}")
            continue

        kernel = _var_match_scale_kernel(stim, np.asarray(kernel))

        # 旧 filter_plot.py と同じく same convolution を使う。
        # ここでは非線形関数の形状を見ることが主目的。
        g = np.convolve(stim, kernel, "same")

        # BaccusModel.py の実行時に合わせ、g は標準偏差で割る。
        # ただし mean=0 への再オフセットや max abs 正規化は行わない。
        g_std = np.std(g)
        if g_std > 1e-12:
            g = g / g_std

        try:
            u_raw = N_LNK.main(g, a, kappa, b1, b2, ka)
        except Exception as e:
            print(f"[WARN] nonlinear reconstruction failed: {sd}: {e}")
            continue

        n = min(len(g), len(u_raw))

        if n < 10:
            continue

        g = g[:n]
        u_raw = u_raw[:n]

        # 平均シフトなしの 0-1 正規化
        u_norm_0_1 = _normalize_0_1(u_raw)

        seed_data.append(
            dict(
                seed=seed,
                corr=corr,
                g=g,
                u_raw=u_raw,
                u_norm_0_1=u_norm_0_1,
                a=a,
                kappa=kappa,
                b1=b1,
                b2=b2,
                ka=ka,
            )
        )

    if not seed_data:
        print(f"[WARN] no valid seed data: {base_dir}")
        return

    seed_data = sorted(seed_data, key=lambda x: x["corr"])
    best = seed_data[-1]

    out_dir = os.path.join(base_dir, "filter_plot4")
    os.makedirs(out_dir, exist_ok=True)

    roi_label = ROI_LABELS.get(roi, f"ROI{roi}")

    # 1. raw nonlinear output
    plot_nonlinear_overlay(
        seed_data=seed_data,
        out_path=os.path.join(out_dir, "nonlinear_raw_output.pdf"),
        title=f"Raw nonlinear output {roi_label}",
        y_mode="raw",
        max_points=max_points,
    )

    plot_nonlinear_best(
        best=best,
        out_path=os.path.join(out_dir, "best_nonlinear_raw_output.pdf"),
        title=f"Best raw nonlinear output {roi_label}",
        y_mode="raw",
        max_points=max_points,
    )

    # 2. normalized 0-1 nonlinear output
    plot_nonlinear_overlay(
        seed_data=seed_data,
        out_path=os.path.join(out_dir, "nonlinear_normalized_0_1.pdf"),
        title=f"Normalized nonlinear output {roi_label}",
        y_mode="normalized",
        max_points=max_points,
    )

    plot_nonlinear_best(
        best=best,
        out_path=os.path.join(out_dir, "best_nonlinear_normalized_0_1.pdf"),
        title=f"Best normalized nonlinear output {roi_label}",
        y_mode="normalized",
        max_points=max_points,
    )

    # diagnostics
    save_nonlinear_diagnostics(
        seed_data=seed_data,
        out_path=os.path.join(out_dir, "nonlinear_diagnostics.csv"),
    )

    print(f"=== filter_plot4 DONE for ROI {roi} ===")
    print(f"  base_dir : {base_dir}")
    print(f"  out_dir  : {out_dir}")
    print(f"  best seed: {best['seed']} corr={best['corr']:.6f}")
    print(
        f"  best raw u range: "
        f"{np.min(best['u_raw']):.6e} to {np.max(best['u_raw']):.6e}"
    )


# ==========================================================
# main
# ==========================================================

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
        help="目的関数ディレクトリ名 default: band_full",
    )
    parser.add_argument(
        "--roi",
        type=int,
        default=None,
        help="特定ROIのみ処理する場合に指定",
    )
    parser.add_argument(
        "--stim",
        type=str,
        default="data/ret2p/chirp_stim_64Hz_bilinear.txt",
        help="入力刺激ファイル",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.015625,
        help="sampling interval [s]",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="filter window [s]",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=80000,
        help="scatter plot に使う最大点数",
    )
    
    parser.add_argument(
        "--g_min",
        type=float,
        default=-16.0,
    )

    parser.add_argument(
        "--g_max",
        type=float,
        default=16.0,
    )

    args = parser.parse_args()

    root_dir = _safe_path_arg(args.root_dir)

    if not os.path.isdir(root_dir):
        print(f"ERROR: root_dir not found: {root_dir}")
        sys.exit(1)

    if args.roi is None:
        roi_list = list(range(1, 15))
    else:
        if args.roi < 1 or args.roi > 14:
            print("ERROR: --roi must be in 1..14")
            sys.exit(1)
        roi_list = [args.roi]

    for roi in roi_list:
        base_dir = os.path.join(root_dir, f"roi_{roi}", args.objective)

        print("----------------------------------------")
        print(f"[RUN] ROI {roi}")
        print(f"base_dir = {base_dir}")

        run_for_roi(
            base_dir=base_dir,
            roi=roi,
            stim_path=args.stim,
            dt=args.dt,
            tau=args.tau,
            max_points=args.max_points,
        )


if __name__ == "__main__":
    main()