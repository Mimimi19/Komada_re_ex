# src/tools/linear_scatter.py
# -*- coding: utf-8 -*-
"""
Linear filter の特徴点を ROI/subtype ごとにまとめて PDF 保存するツール。

目的:
  各 ROI の best seed（correlation 最大）について、線形フィルタから以下を抽出して図示する。

  - delta:
      星印 (*) で表示。
      ただし delta >= --raw_delta_threshold の ROI は
      「遅延が分解能以下で検出できていない可能性が高い」とみなし、
      delta除去後ではなく raw kernel を用いて特徴点を抽出する。
      raw を使った ROI は print で明示する。

  - 上に凸のポイント:
      前後の点より大きい局所ピーク。
      上三角 (^) で表示。

  - 下に凸のポイント:
      前後の点より小さい局所谷。
      下三角 (v) で表示。

  - 収束したポイント:
      max(|kernel|) に対して一定比率以下の振幅が
      指定点数以上続く最初の時刻。
      丸 (o) で表示。

注意:
  - 「凸」は最大/最小値そのものではなく、前後点に対する局所的な上/下。
  - 初期値点が最大/最小でも、0秒の特徴点は図示しない。
  - delta >= threshold の ROI だけ raw kernel を使い、それ以外は delta を省いた kernel を使う。

実行例:
  uv run python src/tools/linear_scatter.py \
      --base scripts/spring04/scripts/limit \
      --roi "1,2,3,4,5,6,7,8,9,10,11,12,13,14"

出力:
  <base>/linear_scatter/
    linear_feature_times.pdf              # delta を除いた特徴点図
    linear_delta_timeline.pdf             # delta 順の数直線図
    linear_feature_and_delta_summary.pdf  # 左:特徴点図 / 右:delta順図
    linear_feature_times.csv
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


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)

if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

import components.L_LNK as L_LNK


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


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


ROI_COLORS = {
    1:  "navy",
    2:  "blue",
    3:  "royalblue",
    4:  "deepskyblue",
    5:  "cyan",
    6:  "darkred",
    7:  "red",
    8:  "orangered",
    9:  "darkorange",
    10: "orange",
    11: "gold",
    12: "tomato",
    13: "salmon",
    14: "black",
}


def _read_float(path: str):
    try:
        s = open(path, "r", encoding="utf-8").read().strip()
        for tok in s.replace(",", " ").split():
            try:
                return float(tok)
            except Exception:
                pass
        return float(s)
    except Exception:
        return None


def _read_param(seed_dir: str, name: str):
    path = os.path.join(seed_dir, f"{name}.txt")
    if not os.path.exists(path):
        return None

    v = _read_float(path)
    if v is None or not np.isfinite(v):
        return None

    return float(v)


def _read_corr(seed_dir: str):
    return _read_param(seed_dir, "correlation")


def _read_Ls(seed_dir: str):
    vals = []

    for i in range(1, 200):
        path = os.path.join(seed_dir, f"L{i}.txt")

        if not os.path.exists(path):
            break

        v = _read_float(path)

        if v is None or not np.isfinite(v):
            break

        vals.append(float(v))

    if len(vals) == 0:
        return None

    return np.asarray(vals, dtype=float)


def find_best_seed_dir(roi_objective_dir: str):
    seed_dirs = sorted(glob.glob(os.path.join(roi_objective_dir, "seed_*")))

    best_dir = None
    best_corr = -np.inf

    for sd in seed_dirs:
        corr = _read_corr(sd)

        if corr is None:
            continue

        if corr > best_corr:
            best_corr = corr
            best_dir = sd

    if best_dir is None:
        return None, None

    return best_dir, float(best_corr)


def reconstruct_kernels(alphas, delta: float, dt: float, tau: float):
    """
    filter_plot2.py と同じ方針で、L_LNK.main() 由来の kernel を再構成する。
    """
    filter_points = int(tau / dt) + 1

    kernel_conv, _ = L_LNK.main(alphas, delta, filter_points, dt, tau)

    kernel_plot_raw = np.asarray(kernel_conv, dtype=float)[::-1]
    axis_raw = np.arange(len(kernel_plot_raw)) * dt

    delta_idx = int(round(delta / dt))
    delta_idx = max(0, min(delta_idx, len(kernel_plot_raw) - 1))

    kernel_plot_delay_removed = kernel_plot_raw[delta_idx:]
    axis_delay_removed = np.arange(len(kernel_plot_delay_removed)) * dt

    return {
        "kernel_plot_raw": kernel_plot_raw,
        "kernel_plot_delay_removed": kernel_plot_delay_removed,
        "axis_raw": axis_raw,
        "axis_delay_removed": axis_delay_removed,
    }


def normalize_for_feature_detection(k: np.ndarray):
    k = np.asarray(k, dtype=float)

    if len(k) == 0:
        return k

    k = k - np.mean(k)

    m = np.max(np.abs(k))

    if m > 1e-12:
        k = k / m

    return k


def strongest_local_extremum_time(
    t: np.ndarray,
    k: np.ndarray,
    kind: str,
    min_time: float,
    min_prominence: float,
):
    """
    前後点に対する局所的な上凸/下凸を探し、
    絶対振幅が最も大きい候補の時刻を返す。
    """
    if len(k) < 3:
        return None, None

    candidates = []

    for i in range(1, len(k) - 1):
        if t[i] <= min_time:
            continue

        if kind == "peak":
            is_ext = (k[i] > k[i - 1]) and (k[i] > k[i + 1])
        elif kind == "trough":
            is_ext = (k[i] < k[i - 1]) and (k[i] < k[i + 1])
        else:
            raise ValueError("kind must be 'peak' or 'trough'")

        if not is_ext:
            continue

        local_prominence = abs(k[i] - 0.5 * (k[i - 1] + k[i + 1]))

        if local_prominence < min_prominence:
            continue

        candidates.append((i, abs(k[i])))

    if not candidates:
        return None, None

    best_i = max(candidates, key=lambda x: x[1])[0]

    return float(t[best_i]), float(k[best_i])


def convergence_time(
    t: np.ndarray,
    k: np.ndarray,
    min_time: float,
    threshold_ratio: float,
    stable_points: int,
):
    """
    収束点:
      max(|k|) * threshold_ratio 以下の振幅が stable_points 点以上連続する最初の時刻。
    """
    if len(k) < stable_points + 1:
        return None, None

    max_abs = np.max(np.abs(k))

    if max_abs <= 1e-12:
        return None, None

    threshold = max_abs * threshold_ratio

    for i in range(1, len(k) - stable_points):
        if t[i] <= min_time:
            continue

        window = np.abs(k[i:i + stable_points])

        if np.all(window <= threshold):
            return float(t[i]), float(k[i])

    return None, None


def analyze_best_seed(
    seed_dir: str,
    dt: float,
    tau: float,
    raw_delta_threshold: float,
    min_time: float,
    min_prominence: float,
    convergence_threshold_ratio: float,
    convergence_stable_points: int,
):
    alphas = _read_Ls(seed_dir)
    delta = _read_param(seed_dir, "delta")
    corr = _read_corr(seed_dir)

    if alphas is None or delta is None or corr is None:
        return None

    kernels = reconstruct_kernels(alphas, delta, dt, tau)

    use_raw = bool(delta >= raw_delta_threshold)

    if use_raw:
        t = kernels["axis_raw"]
        k = kernels["kernel_plot_raw"]
    else:
        t = kernels["axis_delay_removed"]
        k = kernels["kernel_plot_delay_removed"]

    k_norm = normalize_for_feature_detection(k)

    peak_t, peak_amp = strongest_local_extremum_time(
        t=t,
        k=k_norm,
        kind="peak",
        min_time=min_time,
        min_prominence=min_prominence,
    )

    trough_t, trough_amp = strongest_local_extremum_time(
        t=t,
        k=k_norm,
        kind="trough",
        min_time=min_time,
        min_prominence=min_prominence,
    )

    conv_t, conv_amp = convergence_time(
        t=t,
        k=k_norm,
        min_time=min_time,
        threshold_ratio=convergence_threshold_ratio,
        stable_points=convergence_stable_points,
    )

    return {
        "seed": os.path.basename(seed_dir),
        "corr": float(corr),
        "delta": float(delta),
        "use_raw": use_raw,
        "peak_t": peak_t,
        "peak_amp": peak_amp,
        "trough_t": trough_t,
        "trough_amp": trough_amp,
        "conv_t": conv_t,
        "conv_amp": conv_amp,
    }


def collect_linear_features(
    base_dir: str,
    roi_list: list[int],
    objective: str,
    dt: float,
    tau: float,
    raw_delta_threshold: float,
    min_time: float,
    min_prominence: float,
    convergence_threshold_ratio: float,
    convergence_stable_points: int,
):
    rows = []

    for roi in roi_list:
        roi_dir = os.path.join(base_dir, f"roi_{roi}", objective)

        if not os.path.isdir(roi_dir):
            print(f"[WARN] directory not found: {roi_dir}")
            continue

        best_seed_dir, _ = find_best_seed_dir(roi_dir)

        if best_seed_dir is None:
            print(f"[WARN] no valid seed found: ROI {roi}")
            continue

        features = analyze_best_seed(
            seed_dir=best_seed_dir,
            dt=dt,
            tau=tau,
            raw_delta_threshold=raw_delta_threshold,
            min_time=min_time,
            min_prominence=min_prominence,
            convergence_threshold_ratio=convergence_threshold_ratio,
            convergence_stable_points=convergence_stable_points,
        )

        if features is None:
            print(f"[WARN] failed to analyze best seed: {best_seed_dir}")
            continue

        label = ROI_LABELS.get(roi, f"ROI {roi}")

        if features["use_raw"]:
            print(
                f"[RAW] ROI {roi:02d} {label}: "
                f"delta={features['delta']:.6f} >= {raw_delta_threshold:.6f}; "
                f"use raw kernel for feature extraction"
            )

        print(
            f"[OK] ROI {roi:02d} {label}: "
            f"{features['seed']}, corr={features['corr']:.4f}, "
            f"delta={features['delta']:.4f}, "
            f"peak={features['peak_t']}, trough={features['trough_t']}, conv={features['conv_t']}"
        )

        rows.append({
            "roi": roi,
            "label": label,
            "color": ROI_COLORS.get(roi, "gray"),
            **features,
        })

    return rows


def _plot_feature_marker(ax, x, y, marker, color, text_dx=0.04):
    if y is None:
        return

    ax.scatter(
        x,
        y,
        marker=marker,
        s=95,
        color=color,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )

    ax.text(
        x + text_dx,
        y,
        f"{y:.3f}s",
        fontsize=10,
        color=color,
        ha="left",
        va="center",
        zorder=4,
    )


def plot_feature_times_on_axis(ax, rows: list[dict]):
    """
    delta を除いた linear filter 特徴点図を指定 ax に描画する。
    """
    x_positions = np.arange(len(rows))

    for x, r in zip(x_positions, rows):
        color = r["color"]
        
        feature_times = [
            r["peak_t"],
            r["trough_t"],
            r["conv_t"],
        ]

        feature_times = [v for v in feature_times if v is not None]

        if len(feature_times) >= 2:
            ax.plot(
                [x, x],
                [min(feature_times), max(feature_times)],
                color=color,
                linewidth=1.2,
                alpha=0.6,
                zorder=2,
            )

        # delta は別図にまとめるため、この図には描かない
        _plot_feature_marker(ax, x, r["peak_t"], marker="^", color=color)
        _plot_feature_marker(ax, x, r["trough_t"], marker="v", color=color)
        _plot_feature_marker(ax, x, r["conv_t"], marker="o", color=color)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right")

    ax.set_ylabel("Time (s)", fontsize=14)
    ax.set_xlabel("BC subtypes", fontsize=14)
    ax.set_title("Linear filter feature times", fontsize=18)

    ax.grid(True, axis="y", alpha=0.3)

    # 凡例がデータと被らないように、縦軸上限へ 0.5 秒の余白を追加
    y_values = []
    for r in rows:
        for key in ["peak_t", "trough_t", "conv_t"]:
            if r.get(key) is not None:
                y_values.append(r[key])

    if y_values:
        ymin_current, ymax_current = ax.get_ylim()
        ymax_data = max(y_values)
        ax.set_ylim(ymin_current, max(ymax_current, ymax_data + 0.05))

    handles = [
        plt.Line2D([0], [0], marker="^", linestyle="None", color="black",
                   markerfacecolor="black", markersize=8, label="local upward point"),
        plt.Line2D([0], [0], marker="v", linestyle="None", color="black",
                   markerfacecolor="black", markersize=8, label="local downward point"),
        plt.Line2D([0], [0], marker="o", linestyle="None", color="black",
                   markerfacecolor="black", markersize=8, label="convergence point"),
    ]

    ax.legend(bbox_to_anchor=(0.08 , 1), handles=handles, fontsize=10, loc="upper left", borderaxespad=1)


def save_linear_feature_plot(rows: list[dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(12.5, 5.6))

    plot_feature_times_on_axis(ax, rows)

    fig.tight_layout()

    save_path = os.path.join(out_dir, "linear_feature_times.pdf")
    fig.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"[SAVE] {save_path}")


def plot_delta_timeline_on_axis(ax, rows: list[dict]):
    """
    delta の秒数を基準に、細胞名を数直線上へ並べる。
    縦方向は delta 昇順で並べ、横軸が delta [s]。
    """
    # サブタイプ名称順、つまり rows に入っている ROI 順のまま並べる
    sorted_rows = rows
    y_positions = np.arange(len(sorted_rows))

    for y, r in zip(y_positions, sorted_rows):
        color = r["color"]

        ax.scatter(
            r["delta"],
            y,
            marker="o",
            s=80,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        raw_note = " [raw]" if r["use_raw"] else ""

        ax.text(
            r["delta"],
            y,
            f"  {r['label']} \n   ({r['delta']:.3f}s)",
            fontsize=10,
            color=color,
            ha="left",
            va="center",
            zorder=4,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in sorted_rows], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel(r"delay of linear filter $\delta$ (s)", fontsize=14)
    ax.set_title(r"$\delta$ by BC subtype", fontsize=18)
    ax.grid(True, axis="x", alpha=0.3)

    # 数直線っぽく見せる
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_delta_timeline_plot(rows: list[dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(7.5, 6.2))

    plot_delta_timeline_on_axis(ax, rows)

    fig.tight_layout()

    save_path = os.path.join(out_dir, "linear_delta_timeline.pdf")
    fig.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"[SAVE] {save_path}")


def save_feature_and_delta_summary_plot(rows: list[dict], out_dir: str):
    """
    1枚のPDF内に、
      左: deltaを除いた特徴点図
      右: delta順の数直線図
    を配置する。
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17.0, 6.2),
        gridspec_kw={"width_ratios": [1.75, 1.0]},
    )

    plot_feature_times_on_axis(axes[0], rows)
    plot_delta_timeline_on_axis(axes[1], rows)

    fig.tight_layout()

    save_path = os.path.join(out_dir, "linear_feature_and_delta_summary.pdf")
    fig.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"[SAVE] {save_path}")


def save_summary_csv(rows: list[dict], out_dir: str):
    save_path = os.path.join(out_dir, "linear_feature_times.csv")

    header = [
        "roi",
        "label",
        "seed",
        "corr",
        "delta",
        "use_raw",
        "peak_t",
        "peak_amp",
        "trough_t",
        "trough_amp",
        "conv_t",
        "conv_amp",
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")

        for r in rows:
            vals = []

            for h in header:
                v = r.get(h, None)

                if isinstance(v, str):
                    vals.append(f'"{v}"')
                elif isinstance(v, bool):
                    vals.append(str(v))
                elif v is None:
                    vals.append("")
                else:
                    vals.append(f"{v:.10g}")

            f.write(",".join(vals) + "\n")

    print(f"[SAVE] {save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base", required=True, help="例: scripts/spring04/scripts/limit")
    parser.add_argument("--roi", required=True, help='例: "1,2,3,4,5,6,7,8,9,10,11,12,13,14"')
    parser.add_argument("--objective", default="band_full")
    parser.add_argument("--dt", type=float, default=0.015625)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--raw_delta_threshold", type=float, default=0.5)
    parser.add_argument("--min_time", type=float, default=0.0)
    parser.add_argument("--min_prominence", type=float, default=1e-4)
    parser.add_argument("--convergence_threshold_ratio", type=float, default=0.10)
    parser.add_argument("--convergence_stable_points", type=int, default=5)

    args = parser.parse_args()

    roi_list = [int(x) for x in args.roi.replace(" ", "").split(",") if x != ""]

    out_dir = os.path.join(args.base, "linear_scatter")
    os.makedirs(out_dir, exist_ok=True)

    rows = collect_linear_features(
        base_dir=args.base,
        roi_list=roi_list,
        objective=args.objective,
        dt=args.dt,
        tau=args.tau,
        raw_delta_threshold=args.raw_delta_threshold,
        min_time=args.min_time,
        min_prominence=args.min_prominence,
        convergence_threshold_ratio=args.convergence_threshold_ratio,
        convergence_stable_points=args.convergence_stable_points,
    )

    if not rows:
        print("ERROR: 有効な linear features がありません。")
        return

    save_linear_feature_plot(rows, out_dir)
    save_delta_timeline_plot(rows, out_dir)
    save_feature_and_delta_summary_plot(rows, out_dir)
    save_summary_csv(rows, out_dir)


if __name__ == "__main__":
    main()
