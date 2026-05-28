# src/tools/kinetic_box1.py
# -*- coding: utf-8 -*-
"""
各サブタイプごとの kinetics recovery-inactivation balance を箱ひげ図で可視化する。

計算する指標:
    fast balance:
        kfr/ka - kfi/ka

    slow balance:
        ksr/ka - ksi/ka

解釈:
    値 > 0:
        recovery rate が inactivation rate より相対的に大きい

    値 < 0:
        inactivation rate が recovery rate より相対的に大きい

入力ディレクトリ構造:
    <base>/
      roi_1/
        band_full/
          seed_*/
            ka.txt
            kfi.txt
            kfr.txt
            ksi.txt
            ksr.txt
            correlation.txt
      ...
      roi_14/
        band_full/
          seed_*/

実行例:
    uv run python src/tools/kinetic_box1.py \
        --base scripts/spring04/scripts/limit \
        --roi "1,2,3,4,5,6,7,8,9,10,11,12,13,14"

目的関数ディレクトリを変える場合:
    uv run python src/tools/kinetic_box1.py \
        --base scripts/spring04/scripts/limit \
        --roi "1,2,3,4,5" \
        --objective band_low_only

相関上位5 seedだけ使う場合:
    uv run python src/tools/kinetic_box1.py \
        --base scripts/spring04/scripts/limit \
        --roi "1,2,3,4,5,6,7,8,9,10,11,12,13,14" \
        --top-n 5
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


# ==========================================================
# plot settings
# ==========================================================

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
    # OFF
    1:  "navy",
    2:  "blue",
    3:  "royalblue",
    4:  "deepskyblue",
    5:  "cyan",

    # ON
    6:  "darkred",
    7:  "red",
    8:  "orangered",
    9:  "darkorange",
    10: "orange",
    11: "coral",
    12: "tomato",
    13: "salmon",

    # Rod bipolar
    14: "black",
}


# ==========================================================
# utility
# ==========================================================

def _read_float(path: str):
    """
    txt ファイルから最初に読めた float を返す。
    """
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


def read_param(seed_dir: str, name: str):
    """
    seed_dir/name.txt を読む。
    """
    path = os.path.join(seed_dir, f"{name}.txt")

    if not os.path.exists(path):
        return None

    v = _read_float(path)

    if v is None or not np.isfinite(v):
        return None

    return float(v)


def read_corr(seed_dir: str):
    """
    correlation.txt を読む。
    """
    path = os.path.join(seed_dir, "correlation.txt")

    if not os.path.exists(path):
        return None

    v = _read_float(path)

    if v is None or not np.isfinite(v):
        return None

    return float(v)


def collect_seed_kinetics(roi_objective_dir: str):
    """
    1つの ROI/objective ディレクトリ内の seed_* から
    kinetics parameters と correlation を集める。

    Returns
    -------
    rows : list[dict]
    """
    rows = []

    seed_dirs = sorted(glob.glob(os.path.join(roi_objective_dir, "seed_*")))

    for sd in seed_dirs:
        seed_name = os.path.basename(sd)

        corr = read_corr(sd)

        ka = read_param(sd, "ka")
        kfi = read_param(sd, "kfi")
        kfr = read_param(sd, "kfr")
        ksi = read_param(sd, "ksi")
        ksr = read_param(sd, "ksr")

        if any(v is None for v in [ka, kfi, kfr, ksi, ksr]):
            continue

        if ka <= 0:
            continue

        fast_balance = (kfr / ka) - (kfi / ka)
        slow_balance = (ksr / ka) - (ksi / ka)

        row = {
            "seed": seed_name,
            "corr": corr,
            "ka": ka,
            "kfi": kfi,
            "kfr": kfr,
            "ksi": ksi,
            "ksr": ksr,
            "kfi_over_ka": kfi / ka,
            "kfr_over_ka": kfr / ka,
            "ksi_over_ka": ksi / ka,
            "ksr_over_ka": ksr / ka,
            "fast_balance": fast_balance,
            "slow_balance": slow_balance,
        }

        rows.append(row)

    return rows


def collect_all_roi_data(
    base_dir: str,
    roi_list: list[int],
    objective: str,
    top_n: int | None,
):
    """
    ROIごとに seed kinetics を集める。

    top_n が指定されている場合は、correlation 上位 top_n seed のみ使う。
    """
    all_data = {}

    for roi in roi_list:
        roi_objective_dir = os.path.join(base_dir, f"roi_{roi}", objective)

        if not os.path.isdir(roi_objective_dir):
            print(f"[WARN] directory not found: {roi_objective_dir}")
            continue

        rows = collect_seed_kinetics(roi_objective_dir)

        if not rows:
            print(f"[WARN] no valid kinetics data: ROI {roi}")
            continue

        # correlation がある seed を上位にする。
        # corr=None のものは末尾扱い。
        rows = sorted(
            rows,
            key=lambda r: -np.inf if r["corr"] is None else r["corr"],
            reverse=True,
        )

        if top_n is not None:
            rows = rows[:top_n]

        all_data[roi] = rows

        label = ROI_LABELS.get(roi, f"ROI {roi}")
        print(f"[OK] ROI {roi:02d} {label}: {len(rows)} seeds loaded")

    return all_data


# ==========================================================
# plotting
# ==========================================================

def _boxplot_by_roi(
    all_data: dict[int, list[dict]],
    roi_list: list[int],
    value_key: str,
    ylabel: str,
    title: str,
    out_path: str,
    show_points: bool = True,
):
    """
    ROIごとの value_key 分布を箱ひげ図で描画する。
    """
    labels = []
    data = []
    colors = []

    for roi in roi_list:
        rows = all_data.get(roi, [])

        vals = [
            r[value_key]
            for r in rows
            if r.get(value_key) is not None and np.isfinite(r[value_key])
        ]

        if len(vals) == 0:
            continue

        labels.append(ROI_LABELS.get(roi, f"ROI {roi}"))
        data.append(np.asarray(vals, dtype=float))
        colors.append(ROI_COLORS.get(roi, "gray"))

    if len(data) == 0:
        print(f"[WARN] no data for {value_key}")
        return

    fig, ax = plt.subplots(figsize=(12.0, 5.8))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        boxprops=dict(color="black", linewidth=1.0),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # jittered points
    if show_points:
        rng = np.random.default_rng(0)

        for i, vals in enumerate(data, start=1):
            jitter = rng.normal(loc=0.0, scale=0.045, size=len(vals))
            x = i + jitter

            ax.scatter(
                x,
                vals,
                s=18,
                color=colors[i - 1],
                edgecolor="black",
                linewidth=0.25,
                alpha=0.65,
                zorder=3,
            )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=16)

    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"[SAVE] {out_path}")


def save_fast_balance_boxplot(
    all_data: dict[int, list[dict]],
    roi_list: list[int],
    out_dir: str,
    show_points: bool,
):
    save_path = os.path.join(out_dir, "fast_recovery_inactivation_balance_boxplot.pdf")

    _boxplot_by_roi(
        all_data=all_data,
        roi_list=roi_list,
        value_key="fast_balance",
        ylabel=r"$(k_{fr}/k_a) - (k_{fi}/k_a)$",
        title=r"Fast recovery-inactivation balance by subtype",
        out_path=save_path,
        show_points=show_points,
    )


def save_slow_balance_boxplot(
    all_data: dict[int, list[dict]],
    roi_list: list[int],
    out_dir: str,
    show_points: bool,
):
    save_path = os.path.join(out_dir, "slow_recovery_inactivation_balance_boxplot.pdf")

    _boxplot_by_roi(
        all_data=all_data,
        roi_list=roi_list,
        value_key="slow_balance",
        ylabel=r"$(k_{sr}/k_a) - (k_{si}/k_a)$",
        title=r"Slow recovery-inactivation balance by subtype",
        out_path=save_path,
        show_points=show_points,
    )


def save_summary_csv(
    all_data: dict[int, list[dict]],
    roi_list: list[int],
    out_dir: str,
):
    save_path = os.path.join(out_dir, "kinetic_recovery_balance_values.csv")

    header = [
        "roi",
        "label",
        "seed",
        "corr",
        "ka",
        "kfi",
        "kfr",
        "ksi",
        "ksr",
        "kfi_over_ka",
        "kfr_over_ka",
        "ksi_over_ka",
        "ksr_over_ka",
        "fast_balance",
        "slow_balance",
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")

        for roi in roi_list:
            rows = all_data.get(roi, [])
            label = ROI_LABELS.get(roi, f"ROI {roi}")

            for r in rows:
                vals = [
                    roi,
                    f'"{label}"',
                    f'"{r["seed"]}"',
                    r["corr"] if r["corr"] is not None else np.nan,
                    r["ka"],
                    r["kfi"],
                    r["kfr"],
                    r["ksi"],
                    r["ksr"],
                    r["kfi_over_ka"],
                    r["kfr_over_ka"],
                    r["ksi_over_ka"],
                    r["ksr_over_ka"],
                    r["fast_balance"],
                    r["slow_balance"],
                ]

                formatted = []
                for v in vals:
                    if isinstance(v, str):
                        formatted.append(v)
                    else:
                        formatted.append(f"{float(v):.10g}")

                f.write(",".join(formatted) + "\n")

    print(f"[SAVE] {save_path}")


def print_summary(all_data: dict[int, list[dict]], roi_list: list[int]):
    """
    簡単な統計をターミナルに表示する。
    """
    print("\n--- Summary ---")

    for roi in roi_list:
        rows = all_data.get(roi, [])

        if not rows:
            continue

        fast = np.asarray([r["fast_balance"] for r in rows], dtype=float)
        slow = np.asarray([r["slow_balance"] for r in rows], dtype=float)

        label = ROI_LABELS.get(roi, f"ROI {roi}")

        print(
            f"ROI {roi:02d} {label:12s} | "
            f"n={len(rows):2d} | "
            f"fast median={np.median(fast): .4g}, "
            f"slow median={np.median(slow): .4g}"
        )


# ==========================================================
# main
# ==========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        required=True,
        help="例: scripts/limit",
    )
    parser.add_argument(
        "--roi",
        required=True,
        help='例: "1,2,3,4,5,6,7,8,9,10,11,12,13,14"',
    )
    parser.add_argument(
        "--objective",
        default="band_full",
        help="使用する目的関数ディレクトリ default: band_full",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="correlation 上位 N seed のみ使う。未指定なら全 valid seed を使う。",
    )
    parser.add_argument(
        "--hide-points",
        action="store_true",
        help="箱ひげ図上のseedごとの点を非表示にする。",
    )

    args = parser.parse_args()

    base_dir = args.base
    roi_list = [int(x) for x in args.roi.replace(" ", "").split(",") if x != ""]
    objective = args.objective

    out_dir = os.path.join(base_dir, "kinetic_box1")
    os.makedirs(out_dir, exist_ok=True)

    all_data = collect_all_roi_data(
        base_dir=base_dir,
        roi_list=roi_list,
        objective=objective,
        top_n=args.top_n,
    )

    if len(all_data) == 0:
        print("ERROR: 有効な kinetics data がありません。")
        return

    show_points = not args.hide_points

    save_fast_balance_boxplot(
        all_data=all_data,
        roi_list=roi_list,
        out_dir=out_dir,
        show_points=show_points,
    )

    save_slow_balance_boxplot(
        all_data=all_data,
        roi_list=roi_list,
        out_dir=out_dir,
        show_points=show_points,
    )

    save_summary_csv(
        all_data=all_data,
        roi_list=roi_list,
        out_dir=out_dir,
    )

    print_summary(all_data, roi_list)


if __name__ == "__main__":
    main()