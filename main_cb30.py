# -*- coding: utf-8 -*-
"""
main_cb30.py
============
cb1 / cb2 を「1 ROI 相当」として扱い、各 objective で seed を変えて N 回最適化を回すバッチランナー。

実行例:
  uv run python main_cb30.py \
    --data-config config/data/cb1.yaml \
    --objective band_low_only \
    --n-seeds 30 \
    --seed-start 1 \
    --out-root scripts/limit_cb \
    --data-name cb1Hz

  uv run python main_cb30.py \
    --data-config config/data/cb2.yaml \
    --objective band_full \
    --n-seeds 30 \
    --seed-start 1 \
    --out-root scripts/limit_cb \
    --data-name cb2_64Hz

出力:
  <out_root>/<data_name>/<objective>/seed_XX/  (Hydra run dir)
  例: scripts/limit_cb/cb1_64Hz/band_low_only/seed_01/...

注意:
- 既存の src/model/BaccusModel.py が SciPy の differential_evolution に seed を渡していない場合、
  ここで seed を変えても「完全な再現性」は保証されません（ただし試行ごとの初期乱数系列が変わる可能性はあります）。
- 再現性を強く求める場合は、BaccusModel.py 側で differential_evolution(..., seed=opt_cfg.seed) を追加してください。
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_data_config(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "data" not in cfg:
        raise ValueError(f"invalid yaml (need top-level 'data'): {path}")

    data = cfg["data"]
    if not isinstance(data, dict):
        raise ValueError(f"invalid yaml: data must be dict: {path}")

    input_file = data.get("input_file")
    output_file = data.get("output_file")
    dt = data.get("dt")
    name = data.get("name", None)

    if input_file is None or output_file is None or dt is None:
        raise ValueError(f"yaml must include data.input_file, data.output_file, data.dt: {path}")

    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "dt": float(dt),
        "name": str(name) if name is not None else None,
    }


def ensure_parent_dir(fp: str):
    Path(fp).parent.mkdir(parents=True, exist_ok=True)


def run_one_seed(
    *,
    seed: int,
    stim_path: str,
    resp_path: str,
    dt: float,
    out_dir: str,
    objective: str,
    data_name: str,
    dry_run: bool = False,
):
    """
    1 seed 分だけ BaccusModel.py を Hydra run dir を指定して実行。
    - response は out_dir/data/response.txt に保存（BaccusModel の data.output_file を差し替え）
    """
    out_dir = str(Path(out_dir))
    ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))

    resp_out = os.path.join(out_dir, "data", "response.txt")
    ensure_parent_dir(resp_out)

    # NOTE: ここでは「seed」は環境変数として渡す（BaccusModel 側が使っていない場合もある）
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["NUMPY_RANDOM_SEED"] = str(seed)
    env["SCIPY_RANDOM_SEED"] = str(seed)

    cmd = [
        "uv", "run", "python", "src/model/BaccusModel.py",
        f"data.input_file={stim_path}",
        f"data.output_file={resp_out}",
        f"data.name={data_name}",
        f"data.dt={dt}",
        f"hyper_params.objective_type={objective}",
        # もし Hydra 側に optimization.seed があるなら渡しておく（BaccusModel が使っていない場合もある）
        f"optimization.seed={seed}",
        f"hydra.run.dir={out_dir}",
    ]

    print("\n" + "=" * 80)
    print(f"[RUN] seed={seed:02d}")
    print(f"out_dir   : {out_dir}")
    print(f"objective : {objective}")
    print("CMD:", " ".join(cmd))
    print("=" * 80)

    if dry_run:
        return 0

    p = subprocess.run(cmd, env=env)
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description="Run 30 seeds optimization for cb1/cb2 (single ROI).")
    ap.add_argument("--data-config", required=True, help="例: config/data/cb1.yaml")
    ap.add_argument("--objective", required=True, choices=["band_low_only", "band_main_only", "band_full"],
                    help="目的関数モード")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--out-root", required=True, help="例: scripts/limit_cb")
    ap.add_argument("--data-name", default="", help="保存用のデータ名（空ならyamlの data.name を使う。無ければ cbX）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dcfg = load_data_config(args.data_config)
    stim = dcfg["input_file"]
    resp = dcfg["output_file"]
    dt = dcfg["dt"]

    # data_name の決定（優先: CLI > yaml > config filename stem）
    if args.data_name.strip():
        data_name = args.data_name.strip()
    elif dcfg["name"]:
        data_name = dcfg["name"]
    else:
        data_name = Path(args.data_config).stem

    # 入力刺激・応答は「読み取り専用」なので存在チェックだけ
    if not Path(stim).exists():
        raise FileNotFoundError(f"stimulus not found: {stim}")
    if not Path(resp).exists():
        raise FileNotFoundError(f"response not found: {resp}")

    out_root = Path(args.out_root)
    base_dir = out_root / data_name / args.objective
    base_dir.mkdir(parents=True, exist_ok=True)

    # seed ループ
    failed = 0
    for i in range(args.n_seeds):
        seed = args.seed_start + i
        run_dir = base_dir / f"seed_{seed:02d}"
        rc = run_one_seed(
            seed=seed,
            stim_path=stim,
            resp_path=resp,
            dt=dt,
            out_dir=str(run_dir),
            objective=args.objective,
            data_name=data_name,
            dry_run=args.dry_run,
        )
        if rc != 0:
            failed += 1
            print(f"[WARN] seed {seed:02d} failed with returncode={rc}")

    print("\n=== main_cb30 done ===")
    print(f"data_config : {args.data_config}")
    print(f"data_name   : {data_name}")
    print(f"objective   : {args.objective}")
    print(f"out_dir     : {base_dir}")
    print(f"seeds       : {args.seed_start} .. {args.seed_start + args.n_seeds - 1}")
    if failed:
        print(f"FAILED seeds: {failed}/{args.n_seeds}")
        sys.exit(2)


if __name__ == "__main__":
    main()
