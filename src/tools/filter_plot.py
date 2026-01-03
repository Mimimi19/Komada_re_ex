# src/tools/filter_plot.py
# -*- coding: utf-8 -*-
"""
Seedごとの最適化結果から、以下を可視化するツール。

(1) Linear filter kernel (論文Fig.3A形式):
    - L_LNK.main(...) から返る linear_filter_kernel をそのまま描く
    - 横軸: delay [0..tau] (s)
    - 縦軸: kernel amplitude
    - 正規化: Var(g)=Var(s) になるようにスケーリング
        kernel *= sqrt( var(s) / var(g) ) where g = conv(s, kernel, 'same')

(2) Nonlinear stage (BaccusModel と同じ計算):
    - g(t) を計算して横軸、u(t)=N_LNK.main(g(t), ...) を縦軸に scatter
    - 各seedを alpha=0.5 で重ね描き
    - 平均曲線は描かない（要求仕様）

出力先:
    <base_dir>/filter_plot/

実行例:
    uv run python src/tools/filter_plot.py scripts/limit/roi_1/band_low_only
    uv run python src/tools/filter_plot.py -scripts/limit/roi_1/band_low_only
"""
import os
import sys
import glob
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

# ---- src を import path に追加 (src/tools から components を読む) ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/tools
_SRC_DIR  = os.path.dirname(_THIS_DIR)                      # .../src
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

import components.L_LNK as L_LNK
import components.N_LNK as N_LNK

def _mean0_max1(x: np.ndarray) -> np.ndarray:
    """平均0 + max|x|=1 に正規化"""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    m = np.max(np.abs(x))
    if m > 1e-12:
        x = x / m
    return x


def _safe_path_arg(arg: str) -> str:
    if arg.startswith("-") and os.path.exists(arg[1:]):
        return arg[1:]
    return arg

def _load_txt(path: str) -> np.ndarray:
    return np.genfromtxt(path).astype(float)

def _try_load_hydra_cfg(seed_dir: str):
    cfg_path = os.path.join(seed_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None

def _gather_seed_dirs(base_dir: str) -> list[str]:
    seed_dirs = sorted(glob.glob(os.path.join(base_dir, "seed_*")))
    return [d for d in seed_dirs if os.path.isdir(d)]

def _read_param(seed_dir: str, name: str):
    p = os.path.join(seed_dir, f"{name}.txt")
    if not os.path.exists(p):
        return None
    try:
        return float(np.genfromtxt(p))
    except Exception:
        return None

def _read_Ls(seed_dir: str):
    L = []
    for i in range(1, 200):  # 安全側
        p = os.path.join(seed_dir, f"L{i}.txt")
        if not os.path.exists(p):
            break
        try:
            L.append(float(np.genfromtxt(p)))
        except Exception:
            break
    return np.asarray(L, dtype=float) if len(L) else None

def _find_stimulus(seed_dirs: list[str], fallback: str = "data/ret2p/chirp_stim_64Hz_bilinear.txt"):
    # 1) seed の hydra cfg を優先
    for sd in seed_dirs:
        cfg = _try_load_hydra_cfg(sd)
        if isinstance(cfg, dict):
            data = cfg.get("data", {})
            if isinstance(data, dict) and data.get("input_file"):
                ip = data["input_file"]
                if os.path.exists(ip):
                    return ip
                # 相対パス補完（repo root 推測）
                repo_root_guess = os.path.abspath(os.path.join(sd, "..", "..", "..", "..", ".."))
                cand = os.path.join(repo_root_guess, ip)
                if os.path.exists(cand):
                    return cand

    if os.path.exists(fallback):
        return fallback
    return None

def _var_match_scale_kernel(s: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    論文に合わせて Var(g)=Var(s) となるよう kernel をスケーリング。
    g = conv(s, kernel, 'same')
    kernel *= sqrt(var(s)/var(g))
    """
    s = np.asarray(s, float)
    kernel = np.asarray(kernel, float)

    if len(s) < 3 or len(kernel) < 3:
        return kernel

    g = np.convolve(s, kernel, mode="same")
    vs = np.var(s)
    vg = np.var(g)
    if vg <= 1e-12 or vs <= 1e-12:
        return kernel
    return kernel * np.sqrt(vs / vg)

def _plot_kernel_overlay(kernels: list[np.ndarray], delays: np.ndarray, out_path: str, title: str):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)

    for k in kernels:
        if k is None:
            continue
        ax.plot(delays[:len(k)], k, alpha=0.6, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Delay (s) [0..tau]")
    ax.set_ylabel("Filter amplitude")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def _plot_nonlinear_scatter(G_list, U_list, out_path: str, title: str, max_points_per_seed: int = 80000):
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(1, 1, 1)

    for g, u in zip(G_list, U_list):
        if g is None or u is None:
            continue
        n = min(len(g), len(u))
        g = g[:n]
        u = u[:n]
        if n > max_points_per_seed:
            idx = np.random.choice(n, size=max_points_per_seed, replace=False)
            g = g[idx]
            u = u[idx]
        ax.scatter(g, u, s=6, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("Input: g(t)")
    ax.set_ylabel("Output: u(t)")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", nargs="?", default="", help="例: scripts/limit/roi_1/band_low_only")
    parser.add_argument("--base", default="", help="base_dir をオプション指定したい場合")
    parser.add_argument("--tau_list", default="0.15,1.0", help="kernelを描くtau(s)のリスト。例: 0.15,1.0")
    parser.add_argument("--max_points", type=int, default=80000, help="nonlinear scatter の最大点数/seed")
    args = parser.parse_args()

    base_dir = (args.base.strip() or args.base_dir.strip())
    if not base_dir:
        print("ERROR: base_dir が指定されていません。")
        sys.exit(1)

    base_dir = _safe_path_arg(base_dir)
    if not os.path.isdir(base_dir):
        print(f"ERROR: base_dir が存在しません: {base_dir}")
        sys.exit(1)

    seed_dirs = _gather_seed_dirs(base_dir)
    if len(seed_dirs) == 0:
        print(f"ERROR: seed_* が見つかりません: {base_dir}")
        sys.exit(1)

    stim_path = _find_stimulus(seed_dirs)
    if stim_path is None or (not os.path.exists(stim_path)):
        print("ERROR: stimulus が見つかりませんでした。")
        sys.exit(1)

    # dt は hydra cfg の data.dt を優先。無ければ 64Hz の dt を仮定
    dt = None
    for sd in seed_dirs:
        cfg = _try_load_hydra_cfg(sd)
        if isinstance(cfg, dict):
            data = cfg.get("data", {})
            if isinstance(data, dict) and data.get("dt") is not None:
                try:
                    dt = float(data["dt"])
                    break
                except Exception:
                    pass
    if dt is None:
        dt = 0.015625

    # stimulus（var(s) を使うので mean だけ落とす）
    s = _load_txt(stim_path)
    s = s - np.mean(s)

    out_dir = os.path.join(base_dir, "filter_plot")
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------
    # (1) Linear kernel overlay (Fig.3A)
    # -----------------------
    tau_list = []
    for x in args.tau_list.split(","):
        x = x.strip()
        if not x:
            continue
        tau_list.append(float(x))

    kernels_by_tau = {tau: [] for tau in tau_list}

    for sd in seed_dirs:
        alphas = _read_Ls(sd)
        delta = _read_param(sd, "delta")
        if alphas is None or delta is None:
            continue

        for tau in tau_list:
            filter_points = int(tau / dt) + 1
            try:
                kernel, _ = L_LNK.main(alphas, delta, filter_points, dt, tau)
                kernel = np.asarray(kernel, float)
                kernel = _var_match_scale_kernel(s, kernel)
                kernels_by_tau[tau].append(kernel)
            except Exception:
                continue

    for tau in tau_list:
        ks = kernels_by_tau[tau]
        if len(ks) == 0:
            print(f"WARNING: tau={tau} の kernel を生成できた seed がありませんでした。")
            continue
        delays = np.linspace(0.0, tau, num=len(ks[0]))

        _plot_kernel_overlay(
            ks,
            delays,
            out_path=os.path.join(out_dir, f"linear_filter_kernel_tau{tau:.2f}.pdf"),
            title=f"Linear filter  (tau={tau:.2f}s)",
        )

    # -----------------------
    # (2) Nonlinear scatter: g(t) -> u(t) (BaccusModel準拠)
    # -----------------------
    G_list, U_list = [], []

    for sd in seed_dirs:
        cfg = _try_load_hydra_cfg(sd)
        tau_for_g = None
        if isinstance(cfg, dict):
            hp = cfg.get("hyper_params", {})
            if isinstance(hp, dict) and hp.get("tau") is not None:
                try:
                    tau_for_g = float(hp["tau"])
                except Exception:
                    tau_for_g = None
        if tau_for_g is None:
            tau_for_g = 1.0

        alphas = _read_Ls(sd)
        delta = _read_param(sd, "delta")
        a = _read_param(sd, "a")
        kappa = _read_param(sd, "kappa")
        b1 = _read_param(sd, "b1")
        b2 = _read_param(sd, "b2")
        ka = _read_param(sd, "ka")

        if any(v is None for v in [alphas, delta, a, kappa, b1, b2, ka]):
            continue

        filter_points = int(tau_for_g / dt) + 1
        try:
            kernel, _ = L_LNK.main(alphas, delta, filter_points, dt, tau_for_g)
            kernel = np.asarray(kernel, float)
        except Exception:
            continue

        # BaccusModel と同様の shift 補正
        g_full = np.convolve(s, kernel, mode="full")
        shift_idx = int(tau_for_g / dt)
        if len(g_full) >= shift_idx + len(s):
            g = g_full[shift_idx: shift_idx + len(s)]
        else:
            g = g_full[shift_idx:]
            if len(g) < 3:
                continue

        # BaccusModel: std正規化
        g_std = np.std(g)
        if g_std > 1e-12:
            g = g / g_std

        try:
            u = N_LNK.main(g, a, kappa, b1, b2, ka)
            u = np.asarray(u, float)
        except Exception:
            continue

        n = min(len(g), len(u))
        if n < 10:
            continue
        g_n = _mean0_max1(g[:n])
        u_n = _mean0_max1(u[:n])
        G_list.append(g_n)
        U_list.append(u_n)


    if len(G_list) == 0:
        raise RuntimeError("No valid g/u data found (could not reconstruct g(t), u(t) from saved parameters)")

    _plot_nonlinear_scatter(
        G_list,
        U_list,
        out_path=os.path.join(out_dir, "nonlinear_g_vs_u.pdf"),
        title="Nonlinear filter",
        max_points_per_seed=args.max_points,
    )

    print("=== filter_plot done ===")
    print(f"base_dir : {base_dir}")
    print(f"stimulus : {stim_path}")
    print(f"dt       : {dt}")
    print(f"out_dir  : {out_dir}")
    print("saved    : linear_filter_kernel_tau*.pdf, nonlinear_g_vs_u.pdf")


if __name__ == "__main__":
    main()
