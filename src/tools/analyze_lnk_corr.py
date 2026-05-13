# src/tools/analyze_lnk_corr.py


'''
LNK の Spearman 相関のブロックごとの分布を、ROIグループ間で比較するツール。
具体的には、Mann–Whitney U 検定を行い、p値と Cliff's delta を出力する。
実行例:
    uv run python src/tools/analyze_lnk_corr.py \
        "1,2,3,4,5,6" \
        "14,13,12" \
        --base-dir scripts/limit/block_corr
            
'''

import argparse
import pathlib
import re
from typing import Dict, List
import os
import sys
import numpy as np
from scipy import stats


# 解析対象のブロック名（stats.txt の [Block...] と対応）
BLOCK_NAMES = [
    "Block1_step",
    "Block2_F",
    "Block3_A",
    "Block2+3_F+A",
    "All",
]

def cliffs_delta(a_vals: np.ndarray, b_vals: np.ndarray) -> float:
    """
    Cliff's delta を計算する.
    δ = (#(x>y) - #(x<y)) / (n_A * n_B)
    """
    n_a = len(a_vals)
    n_b = len(b_vals)
    count = 0
    for x in a_vals:
        for y in b_vals:
            if x > y:
                count += 1
            elif x < y:
                count -= 1
    return count / (n_a * n_b)


def parse_roi_stats_file(path: pathlib.Path) -> Dict[str, float]:
    """
    roiX_block_corr_stats.txt をパースして、
    各 Block の mean を辞書で返す。

    返り値の例:
    {
        "Block1_step": 0.573120,
        "Block2_F":    0.776327,
        ...
    }
    """
    text = path.read_text()

    block_means: Dict[str, float] = {}

    # [BlockName] ... mean = value を正規表現で拾う
    # 例:
    # [Block1_step]
    #   n      = 30
    #   mean   = 0.573120
    block_pattern = re.compile(r"\[(?P<block>[^\]]+)\]")
    mean_pattern = re.compile(r"mean\s*=\s*([+-]?\d+\.\d+|[+-]?\d+)")

    lines = text.splitlines()
    current_block = None

    for line in lines:
        m_block = block_pattern.search(line)
        if m_block:
            current_block = m_block.group("block").strip()
            continue

        if current_block is not None:
            m_mean = mean_pattern.search(line)
            if m_mean:
                val = float(m_mean.group(1))
                block_means[current_block] = val
                # 次のブロックに移るまでは同じ block とみなす
                # （mean は1回しか出てこないのでこれでOK）

    return block_means


def parse_roi_list(arg: str) -> List[int]:
    """
    "1,2,3,4,5" のような文字列を [1,2,3,4,5] に変換。
    """
    if not arg:
        return []
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def mannwhitney_test(group_a_vals: np.ndarray, group_b_vals: np.ndarray):
    """
    2群の Mann–Whitney U 検定（両側）を行う。
    """
    U, p = stats.mannwhitneyu(group_a_vals, group_b_vals, alternative="two-sided")
    return U, p


def main():
    parser = argparse.ArgumentParser(
        description="Compare LNK Spearman correlations between ROI groups using Mann–Whitney U test."
    )
    parser.add_argument(
        "a",
        type=str,
        help='Group A ROI indices (e.g. "1,2,3,4,5,6")',
    )
    parser.add_argument(
        "b",
        type=str,
        help='Group B ROI indices (e.g. "14,13,12")',
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="scripts/limit/block_corr",
        help="Base directory containing roi_X/roiX_block_corr_stats.txt",
    )
    args = parser.parse_args()

    group_a = parse_roi_list(args.a)
    group_b = parse_roi_list(args.b)

    if not group_a or not group_b:
        raise ValueError("Both group A and group B must have at least one ROI index.")

    base_dir = pathlib.Path(args.base_dir)

    # ROIごとの block → mean を集める
    # 例: roi_means[1]["Block1_step"] = 0.573...
    roi_means: Dict[int, Dict[str, float]] = {}

    def load_roi(roi_idx: int):
        roi_dir = base_dir / f"roi_{roi_idx}"
        stats_path = roi_dir / f"roi{roi_idx}_block_corr_stats.txt"
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found for ROI {roi_idx}: {stats_path}")
        block_means = parse_roi_stats_file(stats_path)
        roi_means[roi_idx] = block_means

    # 全 ROIs を読み込む
    for roi_idx in set(group_a + group_b):
        load_roi(roi_idx)

    print("Group A ROIs:", group_a)
    print("Group B ROIs:", group_b)
    print(f"Base dir: {base_dir}")
    print("-" * 80)

    # 各 Block ごとに A/B の分布を作って Mann–Whitney
# src/tools/analyze_lnk_corr.py
import argparse
import pathlib
import re
from typing import Dict, List
import os
import sys
import numpy as np
from scipy import stats


# 解析対象のブロック名（stats.txt の [Block...] と対応）
BLOCK_NAMES = [
    "Block1_step",
    "Block2_F",
    "Block3_A",
    "Block2+3_F+A",
    "All",
]


def parse_roi_stats_file(path: pathlib.Path) -> Dict[str, float]:
    """
    roiX_block_corr_stats.txt をパースして、
    各 Block の mean を辞書で返す。

    返り値の例:
    {
        "Block1_step": 0.573120,
        "Block2_F":    0.776327,
        ...
    }
    """
    text = path.read_text()

    block_means: Dict[str, float] = {}

    # [BlockName] ... mean = value を正規表現で拾う
    # 例:
    # [Block1_step]
    #   n      = 30
    #   mean   = 0.573120
    block_pattern = re.compile(r"\[(?P<block>[^\]]+)\]")
    mean_pattern = re.compile(r"mean\s*=\s*([+-]?\d+\.\d+|[+-]?\d+)")

    lines = text.splitlines()
    current_block = None

    for line in lines:
        m_block = block_pattern.search(line)
        if m_block:
            current_block = m_block.group("block").strip()
            continue

        if current_block is not None:
            m_mean = mean_pattern.search(line)
            if m_mean:
                val = float(m_mean.group(1))
                block_means[current_block] = val
                # 次のブロックに移るまでは同じ block とみなす
                # （mean は1回しか出てこないのでこれでOK）

    return block_means


def parse_roi_list(arg: str) -> List[int]:
    """
    "1,2,3,4,5" のような文字列を [1,2,3,4,5] に変換。
    """
    if not arg:
        return []
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def mannwhitney_test(group_a_vals: np.ndarray, group_b_vals: np.ndarray):
    """
    2群の Mann–Whitney U 検定（両側）を行う。
    """
    U, p = stats.mannwhitneyu(group_a_vals, group_b_vals, alternative="two-sided")
    return U, p


def main():
    parser = argparse.ArgumentParser(
        description="Compare LNK Spearman correlations between ROI groups using Mann–Whitney U test."
    )
    parser.add_argument(
        "a",
        type=str,
        help='Group A ROI indices (e.g. "1,2,3,4,5,6")',
    )
    parser.add_argument(
        "b",
        type=str,
        help='Group B ROI indices (e.g. "14,13,12")',
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="scripts/limit/block_corr",
        help="Base directory containing roi_X/roiX_block_corr_stats.txt",
    )
    args = parser.parse_args()

    group_a = parse_roi_list(args.a)
    group_b = parse_roi_list(args.b)

    if not group_a or not group_b:
        raise ValueError("Both group A and group B must have at least one ROI index.")

    base_dir = pathlib.Path(args.base_dir)

    # ROIごとの block → mean を集める
    # 例: roi_means[1]["Block1_step"] = 0.573...
    roi_means: Dict[int, Dict[str, float]] = {}

    def load_roi(roi_idx: int):
        roi_dir = base_dir / f"roi_{roi_idx}"
        stats_path = roi_dir / f"roi{roi_idx}_block_corr_stats.txt"
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found for ROI {roi_idx}: {stats_path}")
        block_means = parse_roi_stats_file(stats_path)
        roi_means[roi_idx] = block_means

    # 全 ROIs を読み込む
    for roi_idx in set(group_a + group_b):
        load_roi(roi_idx)

    print("Group A ROIs:", group_a)
    print("Group B ROIs:", group_b)
    print(f"Base dir: {base_dir}")
    print("-" * 80)

    # 各 Block ごとに A/B の分布を作って Mann–Whitney
    # 各 Block ごとに A/B の分布を作って Mann–Whitney
    for block in BLOCK_NAMES:
        # ROIごとの mean を1サンプルとみなす
        a_vals = []
        b_vals = []
        for idx in group_a:
            val = roi_means[idx].get(block, np.nan)
            if not np.isnan(val):
                a_vals.append(val)
        for idx in group_b:
            val = roi_means[idx].get(block, np.nan)
            if not np.isnan(val):
                b_vals.append(val)

        a_vals = np.array(a_vals, dtype=float)
        b_vals = np.array(b_vals, dtype=float)

        if len(a_vals) == 0 or len(b_vals) == 0:
            print(f"[{block}] skipped (no data in one of the groups)")
            continue

        U, p = mannwhitney_test(a_vals, b_vals)
        delta = cliffs_delta(a_vals, b_vals)

        # 効果量のざっくり解釈（Cliff's δ の慣例）
        abs_d = abs(delta)
        if abs_d < 0.147:
            eff_size = "negligible"
        elif abs_d < 0.33:
            eff_size = "small"
        elif abs_d < 0.474:
            eff_size = "medium"
        else:
            eff_size = "large"

        # print(f"[{block}]")
        # print(f"  Group A (ROIs {group_a}): mean = {a_vals.mean():.4f}, n_ROI = {len(a_vals)}")
        # print(f"  Group B (ROIs {group_b}): mean = {b_vals.mean():.4f}, n_ROI = {len(b_vals)}")
        # print(f"  Mann–Whitney U = {U:.2f}, p = {p:.3e}")
        # print(f"  Cliff's delta δ = {delta:.3f} ({eff_size})")
        # print()

        # 出力ディレクトリ
        out_dir = os.path.join(
            base_dir,
            "analyze_lnk_corr",
            f"A_{'_'.join(map(str, group_a))}__B_{'_'.join(map(str, group_b))}",
        )
        os.makedirs(out_dir, exist_ok=True)

        # 統計情報をテキスト出力
        stats_path = os.path.join(out_dir, f"{block}_A_vs_B.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(f"Group A ROIs: {group_a}\n")
            f.write(f"Group B ROIs: {group_b}\n")
            f.write(f"Base dir: {base_dir}\n")
            f.write("-" * 40 + "\n")
            f.write(f"[{block}]\n")
            f.write(f"Group A mean = {a_vals.mean():.4f}, n_ROI = {len(a_vals)}\n")
            f.write(f"Group B mean = {b_vals.mean():.4f}, n_ROI = {len(b_vals)}\n")
            f.write(f"Mann–Whitney U = {U:.2f}, p = {p:.3e}\n")
            f.write(f"Cliff's delta δ = {delta:.3f} ({eff_size})\n")

if __name__ == "__main__":
    main()

