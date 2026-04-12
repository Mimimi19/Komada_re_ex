# -*- coding: utf-8 -*-
"""
main_limit.py (no-repeat version)

前提更新:
- repeat2 のような「同一細胞・同一刺激の繰り返し測定」は存在しない。
  → したがって "noise ceiling (repeat1 vs repeat2)" は直接は測れない。
- ROI番号が異なっていても同一細胞を測っている可能性はあるが、同一細胞の対応付けが
  ファイルとして与えられていない限り、こちらでは同一細胞判定はできない。

本スクリプトの役割:
- ROI平均(roi_1..14)に対して BaccusModel を繰り返し最適化し、"到達性能の分布" を出す。
- 目的関数を3種類（band_low_only / band_main_only / band_full）に分けて実行できる。

補助:
- もし「同一細胞グループ」が分かるなら、--same-cell-map を与えて "pseudo ceiling" を算出できる。
  形式: 2列TSV/CSV/space区切り (roi_index, cell_id)
  例:
    1  A
    2  A
    3  B
    4  C
  → 同一 cell_id 内の ROIペア同士の band Spearman を ceiling として集計

実行例:
1) ROI平均 × 目的関数 band_main_only × seed 30回
   uv run python main_limit.py optimize \
     --stim data/ret2p/chirp_stim_64Hz_bilinear.txt \
     --dt 0.015625 \
     --roi-ave-dir data/ret2p/roi_ave \
     --out-root scripts/limit \
     --objective band_full \
     --roi-list "1,14,13,2,12,3,11,4,10,5,9,6,8,7" \
     --n-seeds 30 \
     --seed-start 1 \
     --data-name ret2p_64Hz
     
     uv run python main_limit.py optimize \
     --stim "data/cb1/wn_0.0002s.txt" \
     --dt 0.015625 \
     --roi-ave-dir data/cb1 \
     --out-root scripts/limit \
     --objective band_low_only \
     --roi-list "[1,14,13,2,12,3,11,4,10,5,9,6,8,7]" \
     --n-seeds 30 \
     --seed-start 1 \
     --data-name ret2p_64Hz

2) (任意) 同一細胞対応がある場合の pseudo ceiling 推定
   uv run python main_limit.py pseudo_ceiling \
     --roi-ave-dir data/ret2p/roi_ave \
     --dt 0.015625 \
     --same-cell-map data/ret2p/same_cell_map.txt \
     --out scripts/results/Baccus_ret2pLimit/pseudo_ceiling.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import spearmanr


# -----------------------------
# band Spearman（補助評価）
# -----------------------------
def _butter_bandpass(x: np.ndarray, dt: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    fs = 1.0 / dt
    nyq = 0.5 * fs
    lo = max(lo, 1e-6)
    hi = min(hi, nyq * 0.999)
    if hi <= lo:
        return x
    b, a = butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return filtfilt(b, a, x)

def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = spearmanr(a, b)
    if np.isnan(rho):
        return 0.0
    return float(rho)

def band_spearman_report(x: np.ndarray, y: np.ndarray, dt: float, order: int = 4) -> Dict[str, float]:
    fs = 1.0 / dt
    nyq = 0.5 * fs
    bands = {
        "low": (0.5, 4.0),
        "main": (4.0, 30.0),
        "high": (30.0, nyq),
        "full_0p5_30": (0.5, 30.0),
    }
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    out: Dict[str, float] = {}
    for k, (lo, hi) in bands.items():
        xb = _butter_bandpass(x, dt, lo, hi, order=order)
        yb = _butter_bandpass(y, dt, lo, hi, order=order)
        out[k] = _safe_spearman(xb, yb)
    return out


# -----------------------------
# 最適化RUNの結果収集
# -----------------------------
def _read_metric(path: Path) -> Optional[float]:
    try:
        txt = path.read_text(encoding="utf-8").strip()
        return float(txt)
    except Exception:
        try:
            arr = np.loadtxt(path)
            if np.ndim(arr) == 0:
                return float(arr)
        except Exception:
            pass
    return None

def _find_best_score(results_dir: Path) -> Optional[float]:
    # priority: final files
    candidates: List[float] = []
    for p in [
        results_dir / "correlation.txt",
        results_dir / "optimal_correlation.txt",
        results_dir / "final_correlation_on_failure.txt",
    ]:
        if p.exists():
            v = _read_metric(p)
            if v is not None:
                candidates.append(v)

    # fallback: last epoch correlation
    epoch_dir = results_dir / "epochs"
    if epoch_dir.exists():
        epoch_files = sorted(epoch_dir.glob("epoch_*_correlation.txt"))
        if epoch_files:
            v = _read_metric(epoch_files[-1])
            if v is not None:
                candidates.append(v)

    if not candidates:
        return None
    return float(candidates[-1])


@dataclass
class OptimizeConfig:
    stim: str
    dt: float
    roi_ave_dir: str
    out_root: str
    objective: str
    n_seeds: int
    seed_start: int
    roi_start: int
    roi_end: int
    roi_list: str = ""
    data_name: str = "ret2pLimit"
    max_workers: int = 4


def _run_one_baccus(stim: str, resp: str, dt: float, out_dir: Path, objective: str, seed: int, data_name: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "python", "src/model/BaccusModel.py",
        f"data.input_file={stim}",
        f"data.output_file={resp}",
        f"data.name={data_name}",
        f"data.dt={dt}",
        f"hyper_params.objective_type={objective}",
        f"optimization.seed={seed}",
        f"optimization.workers=1",
        f"hydra.run.dir={str(out_dir)}",
    ]

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"

    with open(stdout_path, "w", encoding="utf-8") as f_out, open(stderr_path, "w", encoding="utf-8") as f_err:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=f_out,
            stderr=f_err,
            text=True,
        )
    return proc.returncode

def _run_task(task: Dict[str, object]) -> Dict[str, object]:
    roi = int(task["roi"])
    seed = int(task["seed"])
    stim = str(task["stim"])
    resp = str(task["resp"])
    dt = float(task["dt"])
    out_dir = Path(task["out_dir"])
    objective = str(task["objective"])
    data_name = str(task["data_name"])

    try:
        rc = _run_one_baccus(
            stim=stim,
            resp=resp,
            dt=dt,
            out_dir=out_dir,
            objective=objective,
            seed=seed,
            data_name=data_name,
        )
        score = _find_best_score(out_dir)
        return {
            "roi": roi,
            "seed": seed,
            "objective": objective,
            "returncode": rc,
            "metric": score,
            "run_dir": str(out_dir),
            "skipped": False,
            "error": None,
        }
    except Exception as e:
        return {
            "roi": roi,
            "seed": seed,
            "objective": objective,
            "returncode": -1,
            "metric": None,
            "run_dir": str(out_dir),
            "skipped": False,
            "error": f"{type(e).__name__}: {e}",
        }


def optimize(cfg: OptimizeConfig) -> Path:
    roi_dir = Path(cfg.roi_ave_dir)
    out_root = Path(cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    pending_tasks: List[Dict[str, object]] = []

    if getattr(cfg, "roi_list", "") and str(cfg.roi_list).strip() != "":
        roi_list = [int(x) for x in str(cfg.roi_list).split(",") if str(x).strip() != ""]
    else:
        roi_list = list(range(cfg.roi_start, cfg.roi_end + 1))

    start_seed = max(1, int(cfg.seed_start))
    end_seed = int(cfg.n_seeds)

    for roi in roi_list:
        resp_path = roi_dir / f"response_ave_roi{roi}.txt"
        if not resp_path.exists():
            print(f"[SKIP] ROI {roi}: not found {resp_path}")
            continue

        roi_out = out_root / f"roi_{roi}" / cfg.objective
        roi_out.mkdir(parents=True, exist_ok=True)

        for seed in range(start_seed, end_seed + 1):
            run_out = roi_out / f"seed_{seed:02d}"

            if run_out.exists():
                existing = _find_best_score(run_out)
                print(f"[SKIP] exists: {run_out} metric={existing}")
                summary_rows.append({
                    "roi": roi,
                    "objective": cfg.objective,
                    "seed": seed,
                    "returncode": 0,
                    "metric": existing,
                    "run_dir": str(run_out),
                    "skipped": True,
                })
                continue

            pending_tasks.append({
                "roi": roi,
                "seed": seed,
                "stim": cfg.stim,
                "resp": str(resp_path),
                "dt": cfg.dt,
                "out_dir": str(run_out),
                "objective": cfg.objective,
                "data_name": cfg.data_name,
            })

    total = len(pending_tasks)
    done = 0

    print(f"[INFO] queued tasks: {total}")
    print(f"[INFO] max_workers: {cfg.max_workers}")

    with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = [ex.submit(_run_task, task) for task in pending_tasks]

        for fut in as_completed(futures):
            row = fut.result()
            summary_rows.append(row)
            done += 1

            roi = row["roi"]
            seed = row["seed"]
            rc = row["returncode"]
            metric = row["metric"]
            err = row.get("error")

            if err is None:
                print(f"[DONE {done}/{total}] roi={roi} seed={seed} rc={rc} metric={metric}")
            else:
                print(f"[FAIL {done}/{total}] roi={roi} seed={seed} err={err}")

    # ROIごとの統計を再集計
    for roi in roi_list:
        roi_rows = [
            r for r in summary_rows
            if int(r["roi"]) == roi and str(r["objective"]) == cfg.objective and r.get("metric") is not None
        ]
        roi_out = out_root / f"roi_{roi}" / cfg.objective
        roi_out.mkdir(parents=True, exist_ok=True)

        scores = [float(r["metric"]) for r in roi_rows if r["metric"] is not None]

        roi_stats: Dict[str, object] = {
            "roi": roi,
            "objective": cfg.objective,
            "note": "No repeats available; noise ceiling is not measured here.",
        }

        if scores:
            arr = np.asarray(scores, dtype=float)
            roi_stats.update({
                "n_success": int(len(arr)),
                "best": float(np.max(arr)),
                "p95": float(np.quantile(arr, 0.95)),
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
            })
        else:
            roi_stats.update({
                "n_success": 0,
                "best": None,
                "p95": None,
                "median": None,
                "mean": None,
            })

        (roi_out / "roi_stats.json").write_text(
            json.dumps(roi_stats, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    summary_path = out_root / f"summary_{cfg.objective}.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[DONE] summary saved: {summary_path}")
    return summary_path

# -----------------------------
# pseudo ceiling（同一細胞グルーピングがある場合のみ）
# -----------------------------
def _load_same_cell_map(path: str) -> Dict[int, str]:
    """
    2列: roi_index, cell_id
    区切りはカンマ/タブ/スペースのどれでもOK
    """
    m: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # split by common delimiters
            for delim in [",", "\t", " "]:
                if delim in s:
                    parts = [p for p in s.split(delim) if p != ""]
                    break
            else:
                parts = [s]
            if len(parts) < 2:
                continue
            roi = int(parts[0])
            cid = str(parts[1])
            m[roi] = cid
    return m

def pseudo_ceiling(roi_ave_dir: str, dt: float, same_cell_map: str, out: str, order: int = 4) -> Path:
    roi_dir = Path(roi_ave_dir)
    mapping = _load_same_cell_map(same_cell_map)

    # cell_id -> list of roi indices
    groups: Dict[str, List[int]] = {}
    for roi, cid in mapping.items():
        groups.setdefault(cid, []).append(roi)

    pair_reports: List[Dict[str, object]] = []
    for cid, rois in groups.items():
        rois = sorted(rois)
        if len(rois) < 2:
            continue
        # all pairs in this group
        for i in range(len(rois)):
            for j in range(i + 1, len(rois)):
                r1, r2 = rois[i], rois[j]
                p1 = roi_dir / f"response_ave_roi{r1}.txt"
                p2 = roi_dir / f"response_ave_roi{r2}.txt"
                if not (p1.exists() and p2.exists()):
                    continue
                x = np.loadtxt(p1)
                y = np.loadtxt(p2)
                rep = band_spearman_report(x, y, dt=dt, order=order)
                rep.update({"cell_id": cid, "roi1": r1, "roi2": r2})
                pair_reports.append(rep)

    # aggregate
    agg: Dict[str, object] = {
        "dt": dt,
        "same_cell_map": same_cell_map,
        "n_pairs": len(pair_reports),
        "pairs": pair_reports,
    }
    if pair_reports:
        keys = ["low", "main", "high", "full_0p5_30"]
        for k in keys:
            vals = np.array([pr[k] for pr in pair_reports], dtype=float)
            agg[f"{k}_median"] = float(np.median(vals))
            agg[f"{k}_p90"] = float(np.quantile(vals, 0.90))
            agg[f"{k}_p95"] = float(np.quantile(vals, 0.95))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] pseudo ceiling saved: {out_path}")
    return out_path


# -----------------------------
# CLI
# -----------------------------
def _parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_opt = sub.add_parser("optimize")
    p_opt.add_argument("--stim", required=True, help="刺激ファイル（input）")
    p_opt.add_argument("--dt", required=True, type=float, help="サンプリング間隔")
    p_opt.add_argument("--roi-ave-dir", required=True, help="response_ave_roiX.txt のディレクトリ")
    p_opt.add_argument("--out-root", required=True, help="結果保存ルート")
    p_opt.add_argument("--objective", required=True, choices=["band_low_only", "band_main_only", "band_full"])
    p_opt.add_argument("--n-seeds", type=int, default=30)
    p_opt.add_argument("--seed-start", type=int, default=1, help="resume from this seed number (1-based). e.g., --seed-start 7")
    p_opt.add_argument("--roi-start", type=int, default=1)
    p_opt.add_argument("--roi-end", type=int, default=14)
    p_opt.add_argument("--roi-list", type=str, default="", help="Comma separated ROI list (e.g. 14,9,5). If set, overrides --roi-start/--roi-end")
    p_opt.add_argument("--data-name", type=str, default="ret2pLimit")

    p_pc = sub.add_parser("pseudo_ceiling")
    p_pc.add_argument("--roi-ave-dir", required=True)
    p_pc.add_argument("--dt", required=True, type=float)
    p_pc.add_argument("--same-cell-map", required=True, help="2列: roi_index, cell_id")
    p_pc.add_argument("--out", required=True)
    p_pc.add_argument("--order", type=int, default=4)
    p_opt.add_argument("--max-workers", type=int, default=4, help="Number of parallel runs")


    return p.parse_args()


def main():
    args = _parse_args()
    if args.cmd == "optimize":
        cfg = OptimizeConfig(
            stim=args.stim,
            dt=args.dt,
            roi_ave_dir=args.roi_ave_dir,
            out_root=args.out_root,
            objective=args.objective,
            n_seeds=args.n_seeds,
            seed_start=args.seed_start,
            roi_start=args.roi_start,
            roi_end=args.roi_end,
            roi_list=args.roi_list,
            data_name=args.data_name,
            max_workers=args.max_workers,
        )
        optimize(cfg)
    elif args.cmd == "pseudo_ceiling":
        pseudo_ceiling(args.roi_ave_dir, args.dt, args.same_cell_map, args.out, order=args.order)
    else:
        raise RuntimeError("Unknown command")

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
# 重要: BaccusModel.py 側の注意
#
# A) objective_type 3種（band_low_only / band_main_only / band_full）が動く必要あり
#    → obj_hybrid.calculate(...) の内部で band Spearman を重み付きで使う等。
#
# B) differential_evolution に seed を渡すのを強く推奨
#    de_result = differential_evolution(..., seed=opt_cfg.get("seed", None), ...)
# ----------------------------------------------------------------------
