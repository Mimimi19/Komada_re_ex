# main_cb30.py
# -*- coding: utf-8 -*-
"""
cb1 / cb2 の 30 回最適化を回すためのユーティリティ。

- data-config には、
    1) 先頭に data: を持つ Hydra 形式
       data:
         name: cb1
         input_file: ...
         output_file: ...
         dt: 0.0002
    2) いまの cb1.yaml のような top-level 形式
       name: cb1
       input_file: ...
       output_file: ...
       dt: 0.0002
  の両方を受け付ける。

- 各 seed ごとに BaccusModel.py を別ディレクトリに実行する:
    <out_root>/<data_name>/<objective>/seed_XX/

実行例:
    uv run python main_cb30.py \
        --data-config config/data/cb1.yaml \
        --objective band_full \
        --n-seeds 30 \
        --seed-start 1 \
        --out-root scripts/limit_cb \
        --data-name cb1Hz
"""

import os
import sys
import argparse
import subprocess
import yaml


def load_data_config(path: str) -> dict:
    """
    YAML から刺激パス・応答パス・dt を取得する。
    - data: {...} があればその中を使う
    - なければ top-level をそのまま data とみなす
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"data-config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None or not isinstance(cfg, dict):
        raise ValueError(f"invalid yaml: {path}")

    # data: {...} にも top-level にも対応
    if "data" in cfg and isinstance(cfg["data"], dict):
        d = cfg["data"]
    else:
        d = cfg

    required = ["input_file", "output_file", "dt"]
    for k in required:
        if k not in d:
            raise ValueError(f"'{k}' missing in data-config: {path}")

    # name は無くてもよい（あれば使う）
    return d


def build_cmd(
    stim_path: str,
    resp_path: str,
    dt: float,
    data_name: str,
    objective: str,
    seed_dir: str,
    seed: int,
) -> list[str]:
    """
    BaccusModel.py を 1 回叩くコマンドラインを作る。
    """
    cmd = [
        "uv",
        "run",
        "python",
        "src/model/BaccusModel.py",
        f"data.input_file={stim_path}",
        f"data.output_file={resp_path}",
        f"data.name={data_name}",
        f"data.dt={dt}",
        f"hyper_params.objective_type={objective}",
        f"hydra.run.dir={seed_dir}",
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="cb1/cb2 データを 30 回最適化して BaccusModel の限界を測るツール"
    )
    parser.add_argument(
        "--data-config",
        required=True,
        help="cb1.yaml / cb2.yaml などのパス",
    )
    parser.add_argument(
        "--objective",
        required=True,
        choices=["band_low_only", "band_main_only", "band_full"],
        help="目的関数モード（hybrid_objective の objective_mode）",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=30,
        help="試行する seed の個数（デフォルト: 30）",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="seed の開始番号（デフォルト: 1 → 1..n）",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="結果を保存するルートディレクトリ (例: scripts/limit_cb)",
    )
    parser.add_argument(
        "--data-name",
        default="cb_data",
        help="Hydra/BaccusModel に渡す data.name （実験名ラベル）。未指定なら config の name か 'cb_data'",
    )

    args = parser.parse_args()

    dcfg = load_data_config(args.data_config)

    stim_rel = dcfg["input_file"]
    resp_rel = dcfg["output_file"]
    dt = float(dcfg["dt"])
    yaml_name = dcfg.get("name", None)

    # data.name の決定優先度:
    #  CLI --data-name > yaml の name > "cb_data"
    data_name = args.data_name or yaml_name or "cb_data"

    stim_path = stim_rel
    resp_path = resp_rel

    print("=== main_cb30 ===")
    print(f"data-config : {args.data_config}")
    print(f"stim        : {stim_path}")
    print(f"resp        : {resp_path}")
    print(f"dt          : {dt}")
    print(f"data_name   : {data_name}")
    print(f"objective   : {args.objective}")
    print(f"n_seeds     : {args.n_seeds} (start={args.seed_start})")
    print(f"out_root    : {args.out_root}")
    print("--------------")

    # ルートディレクトリ: scripts/limit_cb/<data_name>/<objective>/
    base_out = os.path.join(args.out_root, data_name, args.objective)
    os.makedirs(base_out, exist_ok=True)

    for i in range(args.n_seeds):
        seed = args.seed_start + i
        seed_dir = os.path.join(base_out, f"seed_{seed:02d}")
        os.makedirs(seed_dir, exist_ok=True)

        cmd = build_cmd(
            stim_path=stim_path,
            resp_path=resp_path,
            dt=dt,
            data_name=data_name,
            objective=args.objective,
            seed_dir=seed_dir,
            seed=seed,
        )

        print("\n" + "=" * 80)
        print(f"[RUN] seed={seed}")
        print("CMD: " + " ".join(cmd))
        print("=" * 80)

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] seed={seed} で BaccusModel 実行に失敗しました。")
            print(f"returncode: {e.returncode}")
            # 続行（止めたければ break に変える）
            continue

    print("\n=== main_cb30 DONE ===")


if __name__ == "__main__":
    main()
