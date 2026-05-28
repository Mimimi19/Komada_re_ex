# src/tools/filter_plot2.py
# -*- coding: utf-8 -*-

"""
filter_plot.py の安全版。

目的:
  1. L_LNK.main() が返す「畳み込み用に反転された kernel」を、
     プロット用に戻して描画する。
  2. delta 秒後から図示することで、細胞固有の遅延部分を除外する。
  3. tau をコマンドライン引数で指定可能にする。
  4. x軸を -tau〜0 固定ではなく、正しい dt に基づく時間軸で描く。
  5. QR有無の不一致を確認しやすいように、使用した情報をログに残す。

実行例:
  uv run python src/tools/filter_plot2.py scripts/spring04/scripts/limit/ --objective band_full --tau 1.0
  

特定ROIのみ:
  uv run python src/tools/filter_plot2.py scripts/limit --roi 14 --tau 1.0

0.5秒窓で再構成:
  uv run python src/tools/filter_plot2.py scripts/limit --tau 0.5

注意:
  tau は最適化時と同じ値を指定すること。
  最適化時 tau=1.0 の結果を tau=0.3 で再構成すると、
  保存された L1〜L15 の意味が変わるため、厳密には不整合になる。
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


LEGEND_TOP = 5
RNG = np.random.default_rng(0)

ROI_LABELS = {
    1:  "ROI1 CBC1 (OFF)",
    2:  "ROI2 CBC2 (OFF)",
    3:  "ROI3 CBC3a (OFF)",
    4:  "ROI4 CBC3b (OFF)",
    5:  "ROI5 CBC4 (OFF)",
    6:  "ROI6 CBC5t (ON)",
    7:  "ROI7 CBC5o (ON)",
    8:  "ROI8 CBC5i (ON)",
    9:  "ROI9 CBCX (ON)",
    10: "ROI10 CBC6 (ON)",
    11: "ROI11 CBC7 (ON)",
    12: "ROI12 CBC8 (ON)",
    13: "ROI13 CBC9 (ON)",
    14: "ROI14 RBC (ON)",
}


# ==========================================================
# utility
# ==========================================================

def _safe_path_arg(arg: str) -> str:
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


def _mean0_max1(x):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    m = np.max(np.abs(x))
    if m > 1e-12:
        x = x / m
    return x


def _var_match_scale_kernel_like_model_input(stim, kernel_conv):
    """
    表示用スケーリング。
    形状解釈を壊さないため、必要最小限のスケーリングにする。

    注意:
      これは図示の振幅調整であり、最適化時の目的関数そのものではない。
    """
    g = np.convolve(stim, kernel_conv, mode="same")
    vs = np.var(stim)
    vg = np.var(g)

    if vg <= 1e-12 or vs <= 1e-12:
        return kernel_conv

    return kernel_conv * np.sqrt(vs / vg)


def reconstruct_kernels(alphas, delta, dt, tau):
    """
    L_LNK.main() から得られる畳み込み用 kernel と、
    図示用 kernel を両方返す。

    Returns
    -------
    kernel_conv : np.ndarray
        モデルの畳み込みに使う向きのカーネル。
        L_LNK.main() の返り値そのもの。

    kernel_plot_raw : np.ndarray
        反転を戻した生の図示用カーネル。
        delta を含むため、t=0 は f(delta) になる。

    kernel_plot_delay_removed : np.ndarray
        delta 秒分を除外した図示用カーネル。
        「細胞固有の遅延後を0秒」として見るためのもの。

    axis_raw : np.ndarray
        kernel_plot_raw 用の時間軸。0〜tau。

    axis_delay_removed : np.ndarray
        kernel_plot_delay_removed 用の時間軸。0〜tau-delta。
    """
    filter_points = int(tau / dt + delta) + 1

    # L_LNK.main は畳み込み用に反転して返す
    kernel_conv, _ = L_LNK.main(alphas, delta, filter_points, dt, tau)

    # 図示用に反転を戻す
    kernel_plot_raw = np.asarray(kernel_conv, dtype=float)[::-1]

    axis_raw = np.arange(len(kernel_plot_raw)) * dt

    # delta 秒間を無視する
    delta_idx = int(round(delta / dt))
    delta_idx = max(0, min(delta_idx, len(kernel_plot_raw) - 1))

    kernel_plot_delay_removed = kernel_plot_raw[delta_idx:]
    axis_delay_removed = np.arange(len(kernel_plot_delay_removed)) * dt

    return kernel_conv, kernel_plot_raw, kernel_plot_delay_removed, axis_raw, axis_delay_removed


def compute_g_u_like_optimizer(stim, kernel_conv, a, kappa, b1, b2, ka, dt, tau):
    """
    BaccusModel.py に近い形で g,u を再計算する。

    BaccusModel.py では fftconvolve(..., mode='full') の後、
    shift_idx=int(tau/dt) で切り出している。
    ここでも同じ切り出しを使う。
    """
    from scipy.signal import fftconvolve

    g_full = fftconvolve(stim, kernel_conv, mode="full")
    shift_idx = int(tau / dt)

    if len(g_full) > shift_idx + len(stim):
        g = g_full[shift_idx: shift_idx + len(stim)]
    else:
        g = np.zeros(len(stim))
        take = max(0, min(len(g_full) - shift_idx, len(stim)))
        if take > 0:
            g[:take] = g_full[shift_idx:shift_idx + take]

    g_std = np.std(g)
    if g_std > 1e-12:
        g = g / g_std

    u = N_LNK.main(g, a, kappa, b1, b2, ka)

    n = min(len(g), len(u))
    g = _mean0_max1(g[:n])
    u = _mean0_max1(u[:n])

    return g, u


# ==========================================================
# plotting
# ==========================================================

def plot_kernel_overlay(seed_data, out_path, title, mode="delay_removed"):
    """
    mode:
      delay_removed: delta秒を除外した図示用カーネル
      raw: delta込みの図示用カーネル
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    seed_data_sorted = sorted(seed_data, key=lambda x: x["corr"])
    colors = cm.rainbow(np.linspace(0, 1, len(seed_data_sorted)))

    handles = []

    for i, (d, c) in enumerate(zip(seed_data_sorted, colors)):
        if mode == "raw":
            x = d["axis_raw"]
            k = d["kernel_plot_raw"]
        else:
            x = d["axis_delay_removed"]
            k = d["kernel_plot_delay_removed"]

        label = None
        if i >= len(seed_data_sorted) - LEGEND_TOP:
            label = f"seed {d['seed']} ({d['corr']:.3f}, δ={d['delta']:.3f}s)"

        line = ax.plot(
            x[:len(k)],
            k,
            color=c,
            lw=1.6,
            label=label,
        )

        if label is not None:
            handles.append(line[0])

    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)

    ax.set_title(title)

    if mode == "raw":
        ax.set_xlabel("Time in filter window before latency removal (s)")
    else:
        ax.set_xlabel("Time after latency correction (s)")

    ax.set_ylabel("Linear filter amplitude")
    ax.grid(True, alpha=0.3)

    if handles:
        ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_kernel_best(best, out_path, title, mode="delay_removed", zoom=None):
    fig, ax = plt.subplots(figsize=(5, 3.5))

    if mode == "raw":
        x = best["axis_raw"]
        k = best["kernel_plot_raw"]
    else:
        x = best["axis_delay_removed"]
        k = best["kernel_plot_delay_removed"]

    ax.plot(x[:len(k)], k, lw=2.5, color="black")

    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)

    if zoom is not None:
        ax.set_xlim(zoom)

    ax.set_title(title + f"  (seed {best['seed']} corr={best['corr']:.3f}, δ={best['delta']:.3f}s)")

    if mode == "raw":
        ax.set_xlabel("Time in filter window before latency removal (s)")
    else:
        ax.set_xlabel("Time after latency correction (s)")

    ax.set_ylabel("Linear filter amplitude")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_nonlinear(seed_data, out_path, title, max_points):
    fig, ax = plt.subplots(figsize=(7, 7))

    seed_data_sorted = sorted(seed_data, key=lambda x: x["corr"])
    colors = cm.rainbow(np.linspace(0, 1, len(seed_data_sorted)))

    handles = []

    for i, (d, c) in enumerate(zip(seed_data_sorted, colors)):
        g = d["g"]
        u = d["u"]

        n = min(len(g), len(u))
        g = g[:n]
        u = u[:n]

        if n > max_points:
            idx = RNG.choice(n, max_points, replace=False)
            g = g[idx]
            u = u[idx]

        label = None
        if i >= len(seed_data_sorted) - LEGEND_TOP:
            label = f"seed {d['seed']} ({d['corr']:.3f})"

        h = ax.scatter(
            g,
            u,
            s=6,
            alpha=0.5,
            color=c,
            label=label,
        )

        if label is not None:
            handles.append(h)

    ax.set_title(title)
    ax.set_xlabel("g(t)")
    ax.set_ylabel("u(t)")
    ax.grid(True, alpha=0.3)

    if handles:
        ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_nonlinear_best(best, out_path, title, max_points):
    fig, ax = plt.subplots(figsize=(4, 3))

    g = best["g"]
    u = best["u"]

    n = min(len(g), len(u))
    g = g[:n]
    u = u[:n]

    if n > max_points:
        idx = RNG.choice(n, max_points, replace=False)
        g = g[idx]
        u = u[idx]

    ax.scatter(g, u, s=6, alpha=0.6, color="black")

    ax.set_title(title + f"  (seed {best['seed']} corr={best['corr']:.3f})")
    ax.set_xlabel("g(t)")
    ax.set_ylabel("u(t)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_filter_diagnostics(seed_data, out_path, tau, dt):
    """
    tailがどれくらい残るかを数値で確認する。
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# filter_plot2 diagnostics\n")
        f.write(f"tau = {tau}\n")
        f.write(f"dt = {dt}\n")
        f.write("\n")
        f.write("seed,corr,delta,peak_time_after_delta,tail_abs_mean_0p3_0p6,tail_abs_mean_0p6_end\n")

        for d in sorted(seed_data, key=lambda x: x["corr"], reverse=True):
            k = np.asarray(d["kernel_plot_delay_removed"], dtype=float)
            x = np.asarray(d["axis_delay_removed"], dtype=float)

            if len(k) == 0:
                continue

            peak_time = x[np.argmax(np.abs(k))]

            mask_03_06 = (x >= 0.3) & (x < 0.6)
            mask_06_end = x >= 0.6

            tail_03_06 = np.mean(np.abs(k[mask_03_06])) if np.any(mask_03_06) else np.nan
            tail_06_end = np.mean(np.abs(k[mask_06_end])) if np.any(mask_06_end) else np.nan

            f.write(
                f"{d['seed']},{d['corr']:.6f},{d['delta']:.6f},"
                f"{peak_time:.6f},{tail_03_06:.6e},{tail_06_end:.6e}\n"
            )


# ==========================================================
# core
# ==========================================================

def run_for_roi(base_dir, roi, stim_path, dt, tau, max_points, scale_kernel):
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
        b1 = _read_param(sd, "b1")
        b2 = _read_param(sd, "b2")
        ka = _read_param(sd, "ka")

        # 現在の BaccusModel.py では kappa は最適化対象から外れ、
        # g_t/std 後に kappa=1.0 で N_LNK に入っている。
        kappa = _read_param(sd, "kappa")
        if kappa is None:
            kappa = 1.0

        if alphas is None or any(v is None for v in [delta, a, b1, b2, ka]):
            continue

        try:
            (
                kernel_conv,
                kernel_plot_raw,
                kernel_plot_delay_removed,
                axis_raw,
                axis_delay_removed,
            ) = reconstruct_kernels(alphas, delta, dt, tau)
        except Exception as e:
            print(f"[WARN] kernel reconstruction failed: {sd}: {e}")
            continue

        # 表示振幅を揃えたい場合のみスケーリング。
        # 形そのものを見たいなら --no_scale_kernel を使う。
        if scale_kernel:
            kernel_conv_scaled = _var_match_scale_kernel_like_model_input(stim, kernel_conv)
            scale = np.max(np.abs(kernel_conv_scaled)) / max(np.max(np.abs(kernel_conv)), 1e-12)
            kernel_conv = kernel_conv_scaled
            kernel_plot_raw = kernel_plot_raw * scale
            kernel_plot_delay_removed = kernel_plot_delay_removed * scale

        try:
            g, u = compute_g_u_like_optimizer(
                stim=stim,
                kernel_conv=kernel_conv,
                a=a,
                kappa=kappa,
                b1=b1,
                b2=b2,
                ka=ka,
                dt=dt,
                tau=tau,
            )
        except Exception as e:
            print(f"[WARN] nonlinear reconstruction failed: {sd}: {e}")
            continue

        seed_data.append(
            dict(
                seed=seed,
                corr=corr,
                delta=delta,
                kernel_conv=kernel_conv,
                kernel_plot_raw=kernel_plot_raw,
                kernel_plot_delay_removed=kernel_plot_delay_removed,
                axis_raw=axis_raw,
                axis_delay_removed=axis_delay_removed,
                g=g,
                u=u,
            )
        )

    if not seed_data:
        print(f"[WARN] no valid seed data: {base_dir}")
        return

    seed_data = sorted(seed_data, key=lambda x: x["corr"])
    best = seed_data[-1]

    out_dir = os.path.join(base_dir, "filter_plot2")
    os.makedirs(out_dir, exist_ok=True)

    roi_label = ROI_LABELS.get(roi, f"ROI{roi}")

    # 1. delta込みのraw plot
    plot_kernel_overlay(
        seed_data,
        os.path.join(out_dir, "linear_filter_raw_delta_included.pdf"),
        f"Raw linear filter before latency removal {roi_label}",
        mode="raw",
    )

    plot_kernel_best(
        best,
        os.path.join(out_dir, "best_linear_filter_raw_delta_included.pdf"),
        f"Best raw linear filter before latency removal {roi_label}",
        mode="raw",
    )

    # 2. delta除去後のplot
    plot_kernel_overlay(
        seed_data,
        os.path.join(out_dir, "linear_filter_delay_corrected.pdf"),
        f"Delay-corrected linear filter {roi_label}",
        mode="delay_removed",
    )

    plot_kernel_best(
        best,
        os.path.join(out_dir, "best_linear_filter_delay_corrected.pdf"),
        f"Best delay-corrected linear filter {roi_label}",
        mode="delay_removed",
    )

    # 3. 0〜0.5秒 zoom
    plot_kernel_best(
        best,
        os.path.join(out_dir, "best_linear_filter_delay_corrected_zoom_0_0p5s.pdf"),
        f"Best delay-corrected linear filter zoom {roi_label}",
        mode="delay_removed",
        zoom=(0.0, 0.5),
    )

    # 4. nonlinear
    plot_nonlinear(
        seed_data,
        os.path.join(out_dir, "nonlinear_g_vs_u.pdf"),
        f"Nonlinear {roi_label}",
        max_points,
    )

    plot_nonlinear_best(
        best,
        os.path.join(out_dir, "best_nonlinear_g_vs_u.pdf"),
        f"Best Nonlinear {roi_label}",
        max_points,
    )

    # 5. diagnostics
    save_filter_diagnostics(
        seed_data,
        os.path.join(out_dir, "filter_diagnostics.csv"),
        tau=tau,
        dt=dt,
    )

    print(f"=== filter_plot2 DONE for ROI {roi} ===")
    print(f"  base_dir : {base_dir}")
    print(f"  out_dir  : {out_dir}")
    print(f"  best seed: {best['seed']} corr={best['corr']:.6f} delta={best['delta']:.6f}")


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
        help="filter window [s]. 最適化時と同じtauを指定すること。",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=80000,
    )
    parser.add_argument(
        "--no_scale_kernel",
        action="store_true",
        help="kernel表示用のvariance matching scalingを無効化する",
    )

    args = parser.parse_args()

    root_dir = _safe_path_arg(args.root_dir)

    if not os.path.isdir(root_dir):
        print(f"ERROR: root_dir not found: {root_dir}")
        sys.exit(1)

    if args.roi is None:
        roi_list = list(range(1, 15))
    else:
        roi_list = [args.roi]

    scale_kernel = not args.no_scale_kernel

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
            scale_kernel=scale_kernel,
        )


if __name__ == "__main__":
    main()