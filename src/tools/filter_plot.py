# src/tools/filter_plot.py
# -*- coding: utf-8 -*-
"""
Seedごとの最適化結果から、以下を可視化するツール。

(1) Linear filter kernel (論文Fig.3A形式):
    - L_LNK.main(...) から返る linear_filter_kernel をそのまま描く
    - 横軸: delay [-1.0..0.0] (s)
    - 縦軸: kernel amplitude
    - 正規化: Var(g)=Var(s) になるようにスケーリング
        kernel *= sqrt( var(s) / var(g) ) where g = conv(s, kernel, 'same')

(2) Nonlinear stage (BaccusModel と同じ計算):
    - g(t) を計算して横軸、u(t)=N_LNK.main(g(t), ...) を縦軸に scatter
    - 各seedを alpha=0.5 で重ね描き
    - さらに「相関が最大の seed のみ」を別ファイルに描画

出力先:
    <root_dir>/roi_<ROI>/band_full/filter_plot/

実行例:
    uv run python src/tools/filter_plot.py scripts/limit
"""

import os
import sys
import glob
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント対応
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

# ---- src を import path に追加 (src/tools から components を読む) ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/tools
_SRC_DIR  = os.path.dirname(_THIS_DIR)                      # .../src
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

import components.L_LNK as L_LNK
import components.N_LNK as N_LNK

# ==========================================================
# ROI → ラベル（タイトル・凡例に使う文字列）
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


def _read_corr(seed_dir: str):
    """seed_* ディレクトリ内の correlation.txt を読む（失敗時 None）"""
    p = os.path.join(seed_dir, "correlation.txt")
    if not os.path.exists(p):
        return None
    try:
        v = float(np.genfromtxt(p))
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None


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
    """全 seed のカーネルを重ね描き"""
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)

    for k in kernels:
        if k is None:
            continue
        ax.plot(delays[: len(k)], k, alpha=0.6, linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel("遅延 (s)")
    ax.set_ylabel("フィルタカーネル振幅")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", alpha=0.15)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_kernel_single(kernel: np.ndarray, delays: np.ndarray, out_path: str, title: str):
    """best seed 1本だけのカーネルを描く"""
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(delays[: len(kernel)], kernel, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("遅延 (s)")
    ax.set_ylabel("フィルタカーネル振幅")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", alpha=0.15)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_nonlinear_scatter(G_list, U_list, out_path: str, title: str, max_points_per_seed: int = 80000):
    """全 seed の g-u 散布図を重ね描き"""
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
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_nonlinear_single(g, u, out_path: str, title: str, max_points: int = 80000):
    """best seed 1本だけの g-u 散布図を描く"""
    g = np.asarray(g, float)
    u = np.asarray(u, float)
    n = min(len(g), len(u))
    g = g[:n]
    u = u[:n]
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        g = g[idx]
        u = u[idx]

    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(g, u, s=6, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("Input: g(t)")
    ax.set_ylabel("Output: u(t)")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", alpha=0.15)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# ROI 1つ分を処理するコア関数
# -----------------------------------------------------------------------------
def run_for_one_roi(base_dir: str, tau_list: list[float], max_points: int, roi_id: int | None):
    """
    base_dir: 例 scripts/limit/roi_1/band_full
    tau_list: [0.15, 1.0] など
    max_points: nonlinear scatter の最大点数/seed
    roi_id: int or None
    """
    if not os.path.isdir(base_dir):
        print(f"[WARN] base_dir not found: {base_dir}")
        return

    seed_dirs = _gather_seed_dirs(base_dir)
    if len(seed_dirs) == 0:
        print(f"[WARN] seed_* が見つかりません: {base_dir}")
        return

    stim_path = _find_stimulus(seed_dirs)
    if stim_path is None or (not os.path.exists(stim_path)):
        print(f"[WARN] stimulus が見つかりませんでした: {stim_path}")
        return

    # --- best seed を相関係数から決定 ---
    best_seed_dir = None
    best_seed_id = None
    best_corr = -1e9

    for sd in seed_dirs:
        corr = _read_corr(sd)
        if corr is None:
            continue
        if corr > best_corr:
            best_corr = corr
            best_seed_dir = sd
            try:
                best_seed_id = int(os.path.basename(sd).replace("seed_", ""))
            except Exception:
                best_seed_id = None

    if best_seed_dir is None:
        print(f"[WARN] ROI {roi_id}: correlation.txt を持つ seed がありませんでした。best seed なしで続行します。")

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

    # ROI ラベル
    roi_label = ROI_LABELS.get(roi_id, f"ROI {roi_id}" if roi_id is not None else "")

    # -----------------------
    # (1) Linear kernel overlay (Fig.3A)
    # -----------------------
    kernels_by_tau: dict[float, list[np.ndarray]] = {tau: [] for tau in tau_list}
    best_kernel_by_tau: dict[float, np.ndarray] = {}

    for sd in seed_dirs:
        alphas = _read_Ls(sd)
        delta = _read_param(sd, "delta")
        if alphas is None or delta is None:
            continue

        is_best = (sd == best_seed_dir)

        for tau in tau_list:
            filter_points = int(tau / dt) + 1
            try:
                kernel, _ = L_LNK.main(alphas, delta, filter_points, dt, tau)
                kernel = np.asarray(kernel, float)
                kernel = _var_match_scale_kernel(s, kernel)
                kernels_by_tau[tau].append(kernel)
                if is_best:
                    best_kernel_by_tau[tau] = kernel
            except Exception:
                continue

    for tau in tau_list:
        ks = kernels_by_tau[tau]
        if len(ks) == 0:
            print(f"[WARN] ROI {roi_id}: tau={tau} の kernel を生成できた seed がありませんでした。")
            continue

        # 横軸は [-1.0, 0.0] 固定
        delays = np.linspace(-1.0, 0.0, num=len(ks[0]))

        # ① 全 seed overlay
        _plot_kernel_overlay(
            ks,
            delays,
            out_path=os.path.join(out_dir, f"linear_filter_kernel_tau{tau:.2f}.pdf"),
            title=f"Linear filter {roi_label} (tau={tau:.2f}s, all seeds)",
        )

        # ② best seed のみ
        if tau in best_kernel_by_tau and best_seed_id is not None:
            _plot_kernel_single(
                best_kernel_by_tau[tau],
                delays,
                out_path=os.path.join(out_dir, f"linear_filter_kernel_tau{tau:.2f}_best.pdf"),
                title=f"Linear filter {roi_label} (tau={tau:.2f}s, best seed={best_seed_id:02d}, corr={best_corr:.3f})",
            )

    # -----------------------
    # (2) Nonlinear scatter: g(t) -> u(t) (BaccusModel準拠)
    # -----------------------
    G_list, U_list = [], []
    G_best, U_best = None, None

    for sd in seed_dirs:
        # tau は hydra の hyper_params.tau を優先
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
            tau_for_g = 1.0  # フォールバック

        # ---- パラメータ読み込み ----
        alphas = _read_Ls(sd)
        delta = _read_param(sd, "delta")
        a = _read_param(sd, "a")
        kappa = _read_param(sd, "kappa")
        b1 = _read_param(sd, "b1")
        b2 = _read_param(sd, "b2")
        ka = _read_param(sd, "ka")

        # kappa.txt が無い新しい run では kappa=1.0 に固定
        if kappa is None:
            kappa = 1.0

        # kappa 以外が欠けていたらこの seed はスキップ
        if any(v is None for v in [alphas, delta, a, b1, b2, ka]):
            continue

        # ---- Linear 部分から g(t) を再構成 ----
        filter_points = int(tau_for_g / dt) + 1
        try:
            kernel, _ = L_LNK.main(alphas, delta, filter_points, dt, tau_for_g)
            kernel = np.asarray(kernel, float)
        except Exception:
            continue

        g_full = np.convolve(s, kernel, mode="full")
        shift_idx = int(tau_for_g / dt)

        if len(g_full) >= shift_idx + len(s):
            g = g_full[shift_idx: shift_idx + len(s)]
        else:
            g = g_full[shift_idx:]
            if len(g) < 3:
                continue

        # BaccusModel と同様: g の標準偏差で正規化
        g_std = np.std(g)
        if g_std > 1e-12:
            g = g / g_std

        # ---- Nonlinear: u(t) = N_LNK.main(g, a, kappa, b1, b2, ka) ----
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

        if sd == best_seed_dir:
            G_best = g_n
            U_best = u_n

    # g/u が1本も作れなかった場合：例外にせず Warning を出して終了
    if len(G_list) == 0:
        print(f"[WARN] ROI {roi_id}: g/u を再構成できませんでした。nonlinear プロットをスキップします。")
    else:
        # ① 全 seed
        _plot_nonlinear_scatter(
            G_list,
            U_list,
            out_path=os.path.join(out_dir, "nonlinear_g_vs_u.pdf"),
            title=f"Nonlinear filter {roi_label} (all seeds)",
            max_points_per_seed=max_points,
        )

        # ② best seed のみ
        if G_best is not None and U_best is not None and best_seed_id is not None:
            _plot_nonlinear_single(
                G_best,
                U_best,
                out_path=os.path.join(out_dir, "nonlinear_g_vs_u_best.pdf"),
                title=f"Nonlinear filter {roi_label} (best seed={best_seed_id:02d}, corr={best_corr:.3f})",
                max_points=max_points,
            )

    print(f"=== filter_plot done for ROI {roi_id} ===")
    print(f"  base_dir : {base_dir}")
    print(f"  stimulus : {stim_path}")
    print(f"  dt       : {dt}")
    if best_seed_id is not None:
        print(f"  best seed: seed_{best_seed_id:02d}, corr={best_corr:.4f}")
    else:
        print("  best seed: (none)")
    print(f"  out_dir  : {out_dir}")
    print("  saved    :")
    print("    linear_filter_kernel_tau*.pdf          (all seeds)")
    print("    linear_filter_kernel_tau*_best.pdf     (best seed only)")
    print("    nonlinear_g_vs_u.pdf                   (all seeds)")
    print("    nonlinear_g_vs_u_best.pdf              (best seed only)")


# -----------------------------------------------------------------------------
# エントリポイント
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir",
        help="例: scripts/limit （内部で roi_1〜roi_14/band_full を走査）",
    )
    parser.add_argument(
        "--tau_list",
        default="0.15,1.0",
        help="kernelを描くtau(s)のリスト。例: 0.15,1.0",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=80000,
        help="nonlinear scatter の最大点数/seed",
    )
    args = parser.parse_args()

    root_dir = _safe_path_arg(args.root_dir)
    if not os.path.isdir(root_dir):
        print(f"ERROR: root_dir が存在しません: {root_dir}")
        sys.exit(1)

    # tau_list のパース
    tau_list: list[float] = []
    for x in args.tau_list.split(","):
        x = x.strip()
        if x:
            tau_list.append(float(x))

    if not tau_list:
        print("ERROR: tau_list が空です。例: --tau_list 0.15,1.0")
        sys.exit(1)

    # ROI 1〜14 を順番にまわす
    for roi in range(1, 15):
        base_dir = os.path.join(root_dir, f"roi_{roi}", "band_full")
        print("\n----------------------------------------")
        print(f"[RUN] ROI {roi}  base_dir={base_dir}")
        run_for_one_roi(base_dir, tau_list, args.max_points, roi_id=roi)


if __name__ == "__main__":
    main()
