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
     --out-root scripts/results/Baccus_ret2pLimit \
     --objective band_main_only \
     --n-seeds 30

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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    roi_start: int
    roi_end: int
    data_name: str = "ret2pLimit"


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
        f"hydra.run.dir={str(out_dir)}",
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    print(f"[RUN] seed={seed} objective={objective} out={out_dir}")
    return subprocess.call(cmd, env=env)


def optimize(cfg: OptimizeConfig) -> Path:
    roi_dir = Path(cfg.roi_ave_dir)
    out_root = Path(cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    for roi in range(cfg.roi_start, cfg.roi_end + 1):
        resp_path = roi_dir / f"response_ave_roi{roi}.txt"
        if not resp_path.exists():
            print(f"[SKIP] ROI {roi}: not found {resp_path}")
            continue

        roi_out = out_root / f"roi_{roi}" / cfg.objective
        roi_out.mkdir(parents=True, exist_ok=True)

        scores: List[float] = []
        for s in range(cfg.n_seeds):
            seed = s + 1
            run_out = roi_out / f"seed_{seed:02d}"
            rc = _run_one_baccus(cfg.stim, str(resp_path), cfg.dt, run_out, cfg.objective, seed, cfg.data_name)
            score = _find_best_score(run_out)

            summary_rows.append({
                "roi": roi,
                "objective": cfg.objective,
                "seed": seed,
                "returncode": rc,
                "metric": score,
                "run_dir": str(run_out),
            })
            if score is not None:
                scores.append(float(score))

        # ROIごとの集計（取れた分だけ）
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
            roi_stats.update({"n_success": 0, "best": None, "p95": None, "median": None, "mean": None})

        (roi_out / "roi_stats.json").write_text(json.dumps(roi_stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ROI {roi}] stats saved: {roi_out/'roi_stats.json'}")

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
    p_opt.add_argument("--roi-start", type=int, default=1)
    p_opt.add_argument("--roi-end", type=int, default=14)
    p_opt.add_argument("--data-name", type=str, default="ret2pLimit")

    p_pc = sub.add_parser("pseudo_ceiling")
    p_pc.add_argument("--roi-ave-dir", required=True)
    p_pc.add_argument("--dt", required=True, type=float)
    p_pc.add_argument("--same-cell-map", required=True, help="2列: roi_index, cell_id")
    p_pc.add_argument("--out", required=True)
    p_pc.add_argument("--order", type=int, default=4)

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
            roi_start=args.roi_start,
            roi_end=args.roi_end,
            data_name=args.data_name,
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
