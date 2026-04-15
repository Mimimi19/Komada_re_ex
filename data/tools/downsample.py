# tools/downsample.py
# -*- coding: utf-8 -*-
"""
Hydra用の data config yaml (input_file, output_file, dt を持つ想定) を読み込み、
input/output の txt をターゲット周波数へダウンサンプリングして保存し、
dt とファイルパスを更新した新yamlを出力する。

- 低域のドリフトや高域ノイズ対策として、ダウンサンプリング前にアンチエイリアスLPFを内蔵（scipy.signal.resample_poly）
- 「不要なデータを飛ばしてデータ量を減らす」ため、skip_seconds / max_seconds をオプション化
"""

import os
import argparse
import math
import yaml
import numpy as np

try:
    from scipy.signal import resample_poly
except ImportError as e:
    raise ImportError("scipy が必要です: pip install scipy") from e


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def load_txt_1d(path: str) -> np.ndarray:
    # 大きいtxtでも比較的安定（genfromtxtより速いことが多い）
    return np.loadtxt(path, dtype=np.float64)


def crop_by_seconds(x: np.ndarray, dt: float, skip_seconds: float, max_seconds: float | None) -> np.ndarray:
    n = len(x)
    start = int(round(skip_seconds / dt))
    start = max(0, min(start, n))
    if max_seconds is None or max_seconds <= 0:
        return x[start:]
    length = int(round(max_seconds / dt))
    end = min(n, start + max(1, length))
    return x[start:end]


def resample_to_fs(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    # resample_poly: up/down の有理比で実装。内部でアンチエイリアスが入る（重要）
    # fs_out/fs_in = up/down
    ratio = fs_out / fs_in
    # 近い有理数へ：分母上限を大きめに（fs_inが整数ならdownsample因子が綺麗に出やすい）
    # ただし今回は 5000->500(1/10), 5000->200(1/25) みたいに綺麗なのでOK
    # 一般化のために近似
    max_den = 10_000
    from fractions import Fraction
    frac = Fraction(ratio).limit_denominator(max_den)
    up, down = frac.numerator, frac.denominator

    # 安全: 長さが短すぎる場合
    if len(x) < 10:
        return x.copy()

    y = resample_poly(x, up, down, padtype="line")  # padtype line で端の段差を減らす
    return y.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="入力yaml（data config）パス。例: config/data/cb1.yaml")
    ap.add_argument("--target_fs", type=float, required=True, help="ターゲットサンプリング周波数[Hz]。例: 200 or 500")
    ap.add_argument("--out_dir", default=None, help="出力先ディレクトリ（省略時はyamlの隣に downsampled/ を作る）")
    ap.add_argument("--skip_seconds", type=float, default=0.0, help="冒頭を捨てる秒数（データ量削減用）")
    ap.add_argument("--max_seconds", type=float, default=0.0, help="最大秒数で打ち切り（<=0なら無制限）")
    ap.add_argument("--suffix", default=None, help="出力ファイル名サフィックス（例: fs200）。省略時は fs{int(target_fs)}")
    args = ap.parse_args()

    cfg = load_yaml(args.yaml)

    # 想定: cfg は {"input_file":..., "output_file":..., "dt":...} の直下キー
    # もし {"data":{...}} 形式ならここを適宜変更してください
    if "input_file" not in cfg or "output_file" not in cfg or "dt" not in cfg:
        raise KeyError("yaml に input_file / output_file / dt が見つかりません（この形式に合わせてください）")

    input_path = cfg["input_file"]
    output_path = cfg["output_file"]
    dt_in = float(cfg["dt"])
    fs_in = 1.0 / dt_in

    fs_out = float(args.target_fs)
    dt_out = 1.0 / fs_out

    suffix = args.suffix or f"fs{int(round(fs_out))}"

    # 出力ディレクトリ
    base_dir = os.path.dirname(args.yaml)
    out_dir = args.out_dir or os.path.join(base_dir, "downsampled")
    os.makedirs(out_dir, exist_ok=True)

    # 読み込み
    x_in = load_txt_1d(input_path)
    y_in = load_txt_1d(output_path)

    # 長さ揃え（念のため）
    n = min(len(x_in), len(y_in))
    x_in = x_in[:n]
    y_in = y_in[:n]

    # 不要区間の削減
    max_sec = None if args.max_seconds <= 0 else args.max_seconds
    x_in = crop_by_seconds(x_in, dt_in, args.skip_seconds, max_sec)
    y_in = crop_by_seconds(y_in, dt_in, args.skip_seconds, max_sec)

    # ダウンサンプリング（アンチエイリアス込み）
    x_out = resample_to_fs(x_in, fs_in, fs_out)
    y_out = resample_to_fs(y_in, fs_in, fs_out)

    # 出力
    in_name = os.path.splitext(os.path.basename(input_path))[0]
    out_name = os.path.splitext(os.path.basename(output_path))[0]

    input_out_path = os.path.join(out_dir, f"{in_name}_{suffix}.txt")
    output_out_path = os.path.join(out_dir, f"{out_name}_{suffix}.txt")

    np.savetxt(input_out_path, x_out, fmt="%.8f")
    np.savetxt(output_out_path, y_out, fmt="%.8f")

    # 新yaml（dtとパス差し替え）
    new_cfg = dict(cfg)
    new_cfg["input_file"] = input_out_path
    new_cfg["output_file"] = output_out_path
    new_cfg["dt"] = dt_out

    new_yaml_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(args.yaml))[0]}_{suffix}.yaml")
    save_yaml(new_cfg, new_yaml_path)

    print("=== Downsample Done ===")
    print(f"yaml_in   : {args.yaml}")
    print(f"fs_in     : {fs_in:.3f} Hz (dt={dt_in})")
    print(f"fs_out    : {fs_out:.3f} Hz (dt={dt_out})")
    print(f"skip_sec  : {args.skip_seconds}")
    print(f"max_sec   : {args.max_seconds if args.max_seconds>0 else 'none'}")
    print(f"input_out : {input_out_path}  (n={len(x_out)})")
    print(f"output_out: {output_out_path} (n={len(y_out)})")
    print(f"yaml_out  : {new_yaml_path}")
    print("=======================")


if __name__ == "__main__":
    main()
