# main_new.py (cluster/ROI batch runner)
# ------------------------------------------------------------
# - response_data_repeat_1.txt (lchirp_avg: [samples, rois]) を ROIごとに分割
# - cluster_idx.txt に従い、clusterごとのディレクトリに保存
# - 各 ROI について BaccusModel 最適化を実行し、結果を
#   scripts/results/Baccus_ret2p/cluster_{c}/roi_{i}/
#   に集約
# - 学習後に plot_results.py を自動実行して可視化も保存
#
# 使い方:
#   uv run main_new.py
#   uv run main_new.py --stim data/ret2p/chirp_stim_64Hz_bilinear.txt \
#                  --resp-mat data/ret2p/response_data_repeat_1.txt \
#                  --cluster-idx data/ret2p/cluster_idx.txt \
#                  --dt 0.015625 \
#                  --out-root scripts/results/Baccus_ret2pMat
#
# NOTE:
# - BaccusModel は hydra を使う想定なので、subprocess で
#   `uv run python src/model/BaccusModel.py ... hydra.run.dir=...`
#   を呼びます。
# - response_data_repeat_1.txt の shape は (#samples, #rois) を想定。
#   逆の場合は --transpose を付けてください。
# ------------------------------------------------------------

import argparse
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import yaml
from datetime import datetime

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _save_vector_txt(path: Path, vec: np.ndarray) -> None:
    _ensure_parent(path)
    np.savetxt(path, vec.astype(float), fmt="%.10g")

def _save_data_config_yaml(path: Path, input_file: str, output_file: str, dt: float, name: str) -> None:
    _ensure_parent(path)
    cfg = {
        "input_file": input_file,
        "output_file": output_file,
        "dt": float(dt),
        "name": name,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

def _load_cluster_idx(path: Path) -> np.ndarray:
    # cluster_idx.txt は 1列/改行区切り想定。NaN もあり得る。
    raw = np.genfromtxt(path, dtype=float)
    return raw

def _load_response_matrix(path: Path) -> np.ndarray:
    # txt: 空白/カンマ/タブ混在でも np.genfromtxt が比較的頑健
    mat = np.genfromtxt(path, dtype=float)
    if mat.ndim == 1:
        raise ValueError(f"response matrix seems 1D. file={path}")
    return mat

def run_one_roi(
    *,
    roi_dir: Path,
    stim_file: Path,
    dt: float,
    objective_type: str,
    response_vec: np.ndarray,
    baccus_name: str,
    baccus_script: Path,
    plot_script: Path,
    extra_overrides: list[str],
    dry_run: bool,
) -> None:
    """
    1 ROI を学習→plot まで回す。
    生成物:
      - roi_dir/data_config.yaml
      - roi_dir/data/response.txt
      - roi_dir 以下に hydra.run.dir を固定して BaccusModel の成果物
      - roi_dir/validation/* (plot_results.py が吐く)
    """
    roi_dir.mkdir(parents=True, exist_ok=True)

    # ROI専用データを書き出し
    data_dir = roi_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    resp_path = data_dir / "response.txt"
    _save_vector_txt(resp_path, response_vec)

    data_cfg_path = roi_dir / "data_config.yaml"
    _save_data_config_yaml(
        data_cfg_path,
        input_file=str(stim_file),
        output_file=str(resp_path),
        dt=dt,
        name=baccus_name,
    )

    # BaccusModel 実行（hydra.run.dir を roi_dir に固定）
    # 既存プロジェクトの呼び方に合わせ、data.* を override する
    cmd = [
        "uv", "run", "python", str(baccus_script),
        f"data.input_file={stim_file}",
        f"data.output_file={resp_path}",
        f"data.name={baccus_name}",
        f"data.dt={dt}",
        f"hyper_params.objective_type={objective_type}",
        f"hydra.run.dir={roi_dir}",
    ] + extra_overrides

    print("\n" + "="*80)
    print(f"[RUN] ROI dir: {roi_dir}")
    print("CMD:", " ".join(map(str, cmd)))
    print("="*80)

    if dry_run:
        return

    subprocess.run(cmd, check=True)

    # plot_results.py 実行（結果 dir を渡して自動保存）
    # 既存 plot_results.py は (results_dir, data_config_path) を argv で受ける想定
    plot_cmd = [
        "uv", "run", "python", str(plot_script),
        str(roi_dir),
        str(data_cfg_path),
    ]
    print("[PLOT] CMD:", " ".join(plot_cmd))
    subprocess.run(plot_cmd, check=False)  # plot は失敗しても学習結果は残す

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", default="data/ret2p/chirp_stim_64Hz_bilinear.txt",
                    help="刺激波形 (chirp_stim_*). 1列txt想定")
    ap.add_argument("--resp-mat", default="data/ret2p/response_data_repeat_1.txt",
                    help="lchirp_avg を txt で保存した行列 (#samples, #rois)")
    ap.add_argument("--cluster-idx", default="data/ret2p/cluster_idx.txt",
                    help="cluster_idx を txt で保存した1列データ (#rois)")
    ap.add_argument("--dt", type=float, default=0.015625)
    ap.add_argument("--out-root", default="scripts/results/Baccus_ret2pMat",
                    help="保存ルート。例: scripts/results/Baccus_ret2pMat")
    ap.add_argument("--objective", default="hybrid", choices=["hybrid", "spearman"])
    ap.add_argument("--transpose", action="store_true",
                    help="response_data_repeat_1.txt が (#rois, #samples) なら付ける")
    ap.add_argument("--roi-start", type=int, default=1, help="ROI index (1-based) start")
    ap.add_argument("--roi-end", type=int, default=-1, help="ROI index (1-based) end (inclusive). -1 は最後まで")
    ap.add_argument("--only-cluster", type=str, default="",
                    help="例: '9,10' のように指定するとそのクラスタのみ回す")
    ap.add_argument("--max-rois-per-cluster", type=int, default=-1,
                    help="デバッグ用: 各クラスタで回すROI数を制限。-1は無制限")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--baccus-script", default="src/model/BaccusModel.py")
    ap.add_argument("--plot-script", default="src/plot_module/plot_results.py")
    ap.add_argument("--name", default="ret2pMat", help="data.name に入れる識別名")
    ap.add_argument("--extra-override", action="append", default=[],
                    help="BaccusModel に追加で渡したい hydra override。複数回指定可。例: --extra-override hyper_params.use_I2=false")
    args = ap.parse_args()

    stim_file = Path(args.stim)
    resp_mat_file = Path(args.resp_mat)
    cluster_idx_file = Path(args.cluster_idx)
    out_root = Path(args.out_root)
    baccus_script = Path(args.baccus_script)
    plot_script = Path(args.plot_script)

    if not stim_file.exists():
        raise FileNotFoundError(f"stim not found: {stim_file}")
    if not resp_mat_file.exists():
        raise FileNotFoundError(f"response matrix not found: {resp_mat_file}")
    if not cluster_idx_file.exists():
        raise FileNotFoundError(f"cluster idx not found: {cluster_idx_file}")
    if not baccus_script.exists():
        raise FileNotFoundError(f"BaccusModel.py not found: {baccus_script}")
    if not plot_script.exists():
        print(f"[WARN] plot_results.py not found: {plot_script} (plot will fail)")

    # load
    mat = _load_response_matrix(resp_mat_file)
    if args.transpose:
        mat = mat.T

    cluster_idx = _load_cluster_idx(cluster_idx_file)

    if mat.shape[1] != cluster_idx.shape[0]:
        raise ValueError(
            f"shape mismatch: response_mat has {mat.shape[1]} rois but cluster_idx has {cluster_idx.shape[0]} entries. "
            f"Try --transpose if needed."
        )

    n_samples, n_rois = mat.shape
    print("=== Cluster Batch Runner ===")
    print(f"stim        : {stim_file}")
    print(f"resp_mat    : {resp_mat_file} shape={mat.shape}")
    print(f"cluster_idx : {cluster_idx_file} n={len(cluster_idx)}")
    print(f"dt          : {args.dt}")
    print(f"out_root    : {out_root}")
    print(f"objective   : {args.objective}")
    print("--------------")

    # ROI range (1-based)
    roi_start = max(1, args.roi_start)
    roi_end = n_rois if args.roi_end == -1 else min(n_rois, args.roi_end)

    only_clusters = None
    if args.only_cluster.strip():
        only_clusters = set()
        for x in args.only_cluster.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                only_clusters.add(int(float(x)))
            except Exception:
                pass

    # per-cluster counter
    per_cluster_count: dict[int, int] = {}

    for roi_1based in range(roi_start, roi_end + 1):
        roi0 = roi_1based - 1
        c = cluster_idx[roi0]

        # NaN clusterはスキップ
        if not np.isfinite(c):
            continue

        c_int = int(c)

        if only_clusters is not None and c_int not in only_clusters:
            continue

        # clusterごとの制限
        per_cluster_count.setdefault(c_int, 0)
        if args.max_rois_per_cluster != -1 and per_cluster_count[c_int] >= args.max_rois_per_cluster:
            continue

        per_cluster_count[c_int] += 1

        response_vec = mat[:, roi0]
        # 応答が全NaNならスキップ
        if not np.any(np.isfinite(response_vec)):
            continue
        # NaN を 0 埋め（学習が落ちないように）
        if np.any(~np.isfinite(response_vec)):
            response_vec = np.nan_to_num(response_vec, nan=0.0, posinf=0.0, neginf=0.0)

        roi_dir = out_root / f"cluster_{c_int}" / f"roi_{roi_1based}"

        run_one_roi(
            roi_dir=roi_dir,
            stim_file=stim_file,
            dt=args.dt,
            objective_type=args.objective,
            response_vec=response_vec,
            baccus_name=args.name,
            baccus_script=baccus_script,
            plot_script=plot_script,
            extra_overrides=args.extra_override,
            dry_run=args.dry_run,
        )

    print("\n=== DONE ===")
    print("per-cluster counts:", per_cluster_count)

if __name__ == "__main__":
    main()
