# -*- coding: utf-8 -*-
# data/tools/trim_data.py
"""
timeseries trimming script

config yaml を読み込み
時系列データの一部時間範囲を抽出して
data/ret2p/trim/ に保存する

usage:
    uv run data/tools/trim_data.py \
        --config config/data/ret2p-1.yaml \
        --start 8.5 \
        --end 18.5
"""

import argparse
import os
import numpy as np
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(path):
    return np.loadtxt(path)


def save_data(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, data, fmt="%.6f")


def trim_timeseries(data, start_idx, end_idx):
    return data[start_idx:end_idx]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", type=float, required=False, default=0.0)
    parser.add_argument("--end", type=float, required=False, default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)

    stim_path = cfg["input_file"]
    resp_path = cfg["output_file"]
    dt = 0.015625
    
    stim_path = "data/ret2p/chirp_stim_64Hz_bilinear.txt"
    resp_path = "data/ret2p/response_data_repeat_1.txt"
    dt = 0.015625

    print("=== config ===")
    print("stim:", stim_path)
    print("resp:", resp_path)
    print("dt:", dt)

    # index 計算
    start_idx = int(args.start / dt)
    end_idx = int(args.end / dt) if args.end is not None else None

    print("start_idx:", start_idx)
    print("end_idx:", end_idx)

    # load
    stim = load_data(stim_path)
    resp = load_data(resp_path)

    print("stim shape:", stim.shape)
    print("resp shape:", resp.shape)

    # trim
    stim_trim = trim_timeseries(stim, start_idx, end_idx)
    resp_trim = trim_timeseries(resp, start_idx, end_idx)

    # 出力パス
    stim_name = os.path.basename(stim_path)
    resp_name = os.path.basename(resp_path)

    stim_out = f"data/ret2p/trim/{stim_name}"
    resp_out = f"data/ret2p/trim/{resp_name}"

    save_data(stim_out, stim_trim)
    save_data(resp_out, resp_trim)

    print("=== saved ===")
    print(stim_out)
    print(resp_out)


if __name__ == "__main__":
    main()