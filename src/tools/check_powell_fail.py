#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Powell法とDE法の結果を比較し、失敗したseedをリストアップする
使い方:
    uv run python src/tools/check_powell_fail.py scripts/limit_full/roi_1
'''

import os
import glob

def load_correlation(path):
    try:
        return float(open(path, "r").read().strip())
    except:
        return None

def check_fail(base_dir):
    seed_dirs = sorted(glob.glob(os.path.join(base_dir, "seed_*")))
    fails = []
    lines = []

    for sd in seed_dirs:
        seed = os.path.basename(sd)

        powell_path = os.path.join(sd, "correlation.txt")
        de_path = os.path.join(sd, "epochs", "epoch_600_correlation.txt")

        powell = load_correlation(powell_path)
        de = load_correlation(de_path)

        if powell is None or de is None:
            line = f"{seed}: missing file(s)"
            print(line)
            lines.append(line)
            continue

        status = "OK" if powell > de else "FAIL"
        line = f"{seed}: Powell={powell:.3f}  DE={de:.3f} -> {status}"

        print(line)
        lines.append(line)

        if status == "FAIL":
            fails.append(seed)

    # 結果まとめ
    total = len(seed_dirs)
    fail_count = len(fails)

    lines.append("")
    lines.append("=== RESULT ===")
    lines.append(f"Fail count: {fail_count} / {total}")
    lines.append("=== false seeds ===")

    for s in fails:
        lines.append(s)

    # 保存先
    out_path = os.path.join(base_dir, "false_list.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("base_dir", help="例: scripts/limit_full/roi_1/")
    args = p.parse_args()

    check_fail(args.base_dir)
