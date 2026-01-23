# -*- coding: utf-8 -*-
"""
src/tools/boxplot.py

- scripts/limit_full/roi_1/band_full/seed_01/correlation.txt のようなファイルを集計
- ROI 1〜14 について、seed 分布を箱ひげ図で可視化
- 各 ROI ごとに PNG と LaTeX で貼れる figure .tex を出力

想定ディレクトリ構造:
  scripts/limit_full/
    roi_1/
      band_full/
        seed_01/correlation.txt
        ...
    roi_2/
      band_full/
        ...
    ...
    roi_14/
      band_full/
        ...

使い方:
  # デフォルト(root = scripts/limit_full, ROI 1..14)
  uv run python src/tools/boxplot.py

  # root を変えたい場合
  uv run python src/tools/boxplot.py --root scripts/limit
"""
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント対応
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


# -----------------------------------------------------------------------------
# 設定
# -----------------------------------------------------------------------------
DEFAULT_ROOT = "scripts/limit"   # roi_X/band_full の一つ上
OUT_SUBDIR = "boxplot"

# ROI → ラベル（x軸に表示する文字列）
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



# デフォルトで全 ROI を回す
DEFAULT_ROI_LIST = list(range(1, 15))


# -----------------------------------------------------------------------------
# ユーティリティ
# -----------------------------------------------------------------------------
def _read_float(path: str):
    """correlation.txt から float を読む（"corr: 0.123" 形式も許容）"""
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


def collect_correlations(base_dir: str):
    """
    base_dir/
      seed_01/correlation.txt
      seed_02/correlation.txt
      ...
    を探索して読み取る

    Returns
    -------
    corr : np.ndarray
        読み取り成功した objective 値（seed 順）
    seeds_ok : list[str]
        corr と同じ順番の seed ディレクトリ名（例: 'seed_01'）
    seeds_err : list[tuple[str,str]]
        読み取りに失敗した seed と理由のリスト (seed_name, reason)
    """
    corr = []
    seeds_ok = []
    seeds_err = []

    seed_dirs = sorted(glob.glob(os.path.join(base_dir, "seed_*")))
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        cpath = os.path.join(seed_dir, "correlation.txt")
        if not os.path.exists(cpath):
            seeds_err.append((seed_name, "missing_correlation.txt"))
            continue

        v = _read_float(cpath)
        if v is None:
            seeds_err.append((seed_name, "parse_error"))
            continue
        if not np.isfinite(v):
            seeds_err.append((seed_name, "non_finite"))
            continue

        corr.append(float(v))
        seeds_ok.append(seed_name)

    return np.array(corr, dtype=float), seeds_ok, seeds_err


def save_latex_snippet(out_tex: str, png_name: str, title: str):
    """\\includegraphics で貼れる最小 figure を生成"""
    tex = f"""\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=0.9\\linewidth]{{{png_name}}}
  \\caption{{{title}}}
  \\label{{fig:{os.path.splitext(os.path.basename(png_name))[0]}}}
\\end{{figure}}
"""
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)


# -----------------------------------------------------------------------------
# メイン
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=DEFAULT_ROOT,
        help="roi_X/band_full の一つ上のディレクトリ (default: scripts/limit_full)",
    )
    ap.add_argument(
        "--roi-list",
        type=str,
        default=",".join(str(r) for r in DEFAULT_ROI_LIST),
        help='例: "1,2,3,4" （デフォルト: 1〜14 全て）',
    )
    ap.add_argument(
        "--success-min",
        type=float,
        default=0.1,
        help="成功とみなす objective の下限（これ以下は失敗扱い）",
    )
    args = ap.parse_args()

    root_dir = args.root
    if not os.path.isdir(root_dir):
        raise SystemExit(f"[Error] root_dir not found: {root_dir}")

    roi_list = [
        int(x) for x in args.roi_list.replace(" ", "").split(",") if x != ""
    ]

    print("=== boxplot (per ROI) ===")
    print(f"root_dir : {root_dir}")
    print(f"roi_list : {roi_list}")
    print("-------------------------")

    for roi in roi_list:
        # 対象ディレクトリ: <root>/roi_<roi>/band_full
        base_dir = os.path.join(root_dir, f"roi_{roi}", "band_full")
        if not os.path.isdir(base_dir):
            print(f"[WARN] skip ROI {roi}: base_dir not found: {base_dir}")
            continue

        corr, seeds_ok, seeds_err = collect_correlations(base_dir)
        if corr.size == 0:
            print(f"[WARN] skip ROI {roi}: No correlation.txt under seed_*")
            continue

        out_dir = os.path.join(base_dir, OUT_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)

        best = float(np.max(corr))
        med = float(np.median(corr))
        p95 = float(np.quantile(corr, 0.95))
        mean = float(np.mean(corr))

        # 成功 seed の分離（成功率と成功分布を別指標として出す）
        success_min = float(args.success_min)
        success_mask = corr > success_min
        corr_s = corr[success_mask]

        n_total_dirs = len(sorted(glob.glob(os.path.join(base_dir, "seed_*"))))
        n_valid = int(corr.size)
        n_err = int(len(seeds_err))

        # valid の中で「失敗」（成功閾値未満）になった seed
        fail_seeds = [s for s, v in zip(seeds_ok, corr) if v <= success_min]
        success_seeds = [s for s, v in zip(seeds_ok, corr) if v > success_min]

        n_success = int(len(success_seeds))
        n_fail = int(len(fail_seeds))
        success_rate = (n_success / n_valid) if n_valid else 0.0

        if n_success > 0:
            best_s = float(np.max(corr_s))
            med_s = float(np.median(corr_s))
            p95_s = float(np.quantile(corr_s, 0.95))
            mean_s = float(np.mean(corr_s))
        else:
            best_s = med_s = p95_s = mean_s = float("nan")

        # ------------- 統計の保存 -------------
        stats_path = os.path.join(out_dir, "summary_stats.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            # 全体（失敗混在: 読み取り成功分のみ）
            f.write(f"ROI={roi}\n")
            f.write(f"base_dir={base_dir}\n\n")
            f.write(f"n_total_seed_dirs={n_total_dirs}\n")
            f.write(f"n_valid={n_valid}\n")
            f.write(f"n_err={n_err}\n")
            f.write(f"best_valid={best:.6f}\n")
            f.write(f"mean_valid={mean:.6f}\n")
            f.write(f"median_valid={med:.6f}\n")
            f.write(f"p95_valid={p95:.6f}\n")
            f.write("\n")
            # 成功率と成功分布
            f.write(f"success_min={success_min:.6f}\n")
            f.write(f"n_success={n_success}\n")
            f.write(f"n_fail={n_fail}\n")
            f.write(f"success_rate={success_rate:.6f}\n")
            f.write(f"best_success={best_s:.6f}\n")
            f.write(f"mean_success={mean_s:.6f}\n")
            f.write(f"median_success={med_s:.6f}\n")
            f.write(f"p95_success={p95_s:.6f}\n")
            f.write("\n")

            if seeds_err:
                f.write("[read_errors]\n")
                for seed_name, reason in seeds_err:
                    f.write(f"{seed_name}\t{reason}\n")
                f.write("\n")

            if fail_seeds:
                f.write("[below_success_min]\n")
                for seed_name in fail_seeds:
                    f.write(f"{seed_name}\n")
                f.write("\n")

        # ------------- 箱ひげ図のプロット -------------
        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.add_subplot(111)
        ax.boxplot(corr, vert=True, showmeans=True)

        roi_label = ROI_LABELS.get(roi, f"ROI {roi}")
        ax.set_title(f"ROI {roi}: 最適化後の相関係数の分布", fontsize=14)
        ax.set_ylabel("最適化後の相関係数")
        ax.set_xticks([1])
        ax.set_xticklabels([roi_label])

        txt = (
            f"valid: n={n_valid}  median={med:.3f}  p95={p95:.3f}\n"
            f"success: {n_success}/{n_valid}  median(success)={med_s:.3f}"
        )
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=10)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        png_path = os.path.join(out_dir, "boxplot.png")
        pdf_path = os.path.join(out_dir, "boxplot.pdf")  # 論文用に pdf も

        plt.savefig(png_path, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        # LaTeX snippet（PNG を参照させる）
        tex_path = os.path.join(out_dir, "boxplot.tex")
        save_latex_snippet(
            out_tex=tex_path,
            png_name="boxplot.png",
            title=f"ROI {roi} ({roi_label}): objective distribution across seeds",
        )

        print(f"[OK] ROI {roi}: saved -> {out_dir}")

    print("=== boxplot (per ROI) DONE ===")


if __name__ == "__main__":
    main()
