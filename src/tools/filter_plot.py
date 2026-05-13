# src/tools/filter_plot.py
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

'''
BaccusModel.py の学習結果から、線形フィルタと非線形関数をプロットするツール。
実行例:
    uv run python src/tools/filter_plot.py scripts/spring04/scripts/limit
'''

# ---- path ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)

if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

import components.L_LNK as L_LNK
import components.N_LNK as N_LNK


# ==========================================================
# 設定
# ==========================================================

LEGEND_TOP = 5
RNG = np.random.default_rng(0)

ROI_LABELS = {
    1:"ROI1 CBC1 (OFF)",2:"ROI2 CBC2 (OFF)",3:"ROI3 CBC3a (OFF)",4:"ROI4 CBC3b (OFF)",
    5:"ROI5 CBC4 (OFF)",6:"ROI6 CBC5t (ON)",7:"ROI7 CBC5o (ON)",8:"ROI8 CBC5i (ON)",
    9:"ROI9 CBCX (ON)",10:"ROI10 CBC6 (ON)",11:"ROI11 CBC7 (ON)",12:"ROI12 CBC8 (ON)",
    13:"ROI13 CBC9 (ON)",14:"ROI14 RBC (ON)"
}


# ==========================================================
# utility
# ==========================================================

def _load_txt(path):
    return np.genfromtxt(path).astype(float)


def _read_param(seed_dir,name):
    p=os.path.join(seed_dir,f"{name}.txt")
    if not os.path.exists(p):
        return None
    try:
        return float(np.genfromtxt(p))
    except:
        return None


def _read_Ls(seed_dir):
    L=[]
    for i in range(1,200):
        p=os.path.join(seed_dir,f"L{i}.txt")
        if not os.path.exists(p):
            break
        try:
            L.append(float(np.genfromtxt(p)))
        except:
            break
    return np.asarray(L,float) if L else None


def _read_corr(seed_dir):

    p=os.path.join(seed_dir,"correlation.txt")

    if not os.path.exists(p):
        return None

    try:
        v=float(np.genfromtxt(p))
        if np.isfinite(v):
            return v
    except:
        pass

    return None


def _var_match_scale_kernel(s,kernel):

    g=np.convolve(s,kernel,"same")

    vs=np.var(s)
    vg=np.var(g)

    if vg<=1e-12 or vs<=1e-12:
        return kernel

    return kernel*np.sqrt(vs/vg)


def _mean0_max1(x):

    x=x-np.mean(x)
    m=np.max(np.abs(x))

    if m>1e-12:
        x=x/m

    return x


# ==========================================================
# plotting
# ==========================================================

def plot_kernel_overlay(seed_data,delays,out_path,title):

    fig,ax=plt.subplots(figsize=(11,5))

    # 低→高 の順に描画
    seed_data_sorted = sorted(seed_data,key=lambda x:x["corr"])

    colors=cm.rainbow(np.linspace(0,1,len(seed_data_sorted)))

    handles=[]

    for i,(d,c) in enumerate(zip(seed_data_sorted,colors)):

        k=d["kernel"]

        label=None
        if i >= len(seed_data_sorted)-LEGEND_TOP:
            label=f"seed {d['seed']} ({d['corr']:.3f})"

        line=ax.plot(
            delays[:len(k)],
            k,
            color=c,
            lw=1.6,
            label=label
        )

        if label is not None:
            handles.append(line[0])

    ax.set_title(title)
    ax.set_xlabel("Temporal lag (s)")
    ax.set_ylabel("Linear filter amplitude")
    ax.grid(True)

    if handles:
        ax.legend(handles=handles,fontsize=8,loc="upper left",bbox_to_anchor=(1.02,1))

    fig.tight_layout()
    fig.savefig(out_path,bbox_inches="tight")
    plt.close(fig)
    
def plot_kernel_best(best,delays,out_path,title):

    fig,ax=plt.subplots(figsize=(4,3))

    k=best["kernel"]

    ax.plot(
        delays[:len(k)],
        k,
        lw=3,
        color="black"
    )

    ax.set_title(title + f"  (seed {best['seed']} corr={best['corr']:.3f})")
    ax.set_xlabel("Temporal lag (s)")
    ax.set_ylabel("Linear filter amplitude")

    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path,bbox_inches="tight")
    plt.close(fig)
    
def plot_nonlinear(seed_data,out_path,title,max_points):

    fig,ax=plt.subplots(figsize=(7,7))

    seed_data_sorted = sorted(seed_data,key=lambda x:x["corr"])

    colors=cm.rainbow(np.linspace(0,1,len(seed_data_sorted)))

    handles=[]

    for i,(d,c) in enumerate(zip(seed_data_sorted,colors)):

        g=d["g"]
        u=d["u"]

        n=min(len(g),len(u))

        g=g[:n]
        u=u[:n]

        if n>max_points:
            idx=RNG.choice(n,max_points,replace=False)
            g=g[idx]
            u=u[idx]

        label=None
        if i >= len(seed_data_sorted)-LEGEND_TOP:
            label=f"seed {d['seed']} ({d['corr']:.3f})"

        h=ax.scatter(
            g,u,
            s=6,
            alpha=0.5,
            color=c,
            label=label
        )

        if label is not None:
            handles.append(h)

    ax.set_title(title)
    ax.set_xlabel("g(t)")
    ax.set_ylabel("u(t)")
    ax.grid(True)

    if handles:
        ax.legend(handles=handles,fontsize=8,loc="upper left",bbox_to_anchor=(1.02,1))

    fig.tight_layout()
    fig.savefig(out_path,bbox_inches="tight")
    plt.close(fig)

def plot_nonlinear_best(best,out_path,title,max_points):

    fig,ax=plt.subplots(figsize=(4,3))

    g=best["g"]
    u=best["u"]

    n=min(len(g),len(u))

    g=g[:n]
    u=u[:n]

    if n>max_points:
        idx=RNG.choice(n,max_points,replace=False)
        g=g[idx]
        u=u[idx]

    ax.scatter(g,u,s=6,alpha=0.6,color="black")

    ax.set_title(title + f"  (seed {best['seed']} corr={best['corr']:.3f})")
    ax.set_xlabel("g(t)")
    ax.set_ylabel("u(t)")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path,bbox_inches="tight")
    plt.close(fig)
# ==========================================================
# core
# ==========================================================

def run_for_roi(base_dir,tau_list,max_points,roi):

    seed_dirs=sorted(glob.glob(os.path.join(base_dir,"seed_*")))

    if not seed_dirs:
        return

    stim="data/ret2p/chirp_stim_64Hz_bilinear.txt"

    s=_load_txt(stim)
    s=s-np.mean(s)

    dt=0.015625

    seed_data=[]

    for sd in seed_dirs:

        seed=int(os.path.basename(sd).replace("seed_",""))

        corr=_read_corr(sd)
        if corr is None:
            continue

        alphas=_read_Ls(sd)
        delta=_read_param(sd,"delta")

        a=_read_param(sd,"a")
        kappa=_read_param(sd,"kappa")
        b1=_read_param(sd,"b1")
        b2=_read_param(sd,"b2")
        ka=_read_param(sd,"ka")

        if kappa is None:
            kappa=1.0

        if any(v is None for v in [alphas,delta,a,b1,b2,ka]):
            continue

        tau=1.0
        filter_points=int(tau/dt)+1

        try:
            kernel,_=L_LNK.main(alphas,delta,filter_points,dt,tau)
        except:
            continue

        kernel=_var_match_scale_kernel(s,np.asarray(kernel))

        g=np.convolve(s,kernel,"same")

        g_std=np.std(g)
        if g_std>1e-12:
            g=g/g_std

        try:
            u=N_LNK.main(g,a,kappa,b1,b2,ka)
        except:
            continue

        n=min(len(g),len(u))
        if n<10:
            continue

        g=_mean0_max1(g[:n])
        u=_mean0_max1(u[:n])

        seed_data.append(dict(
            seed=seed,
            corr=corr,
            kernel=kernel,
            g=g,
            u=u
        ))

    if not seed_data:
        return

    seed_data = sorted(seed_data,key=lambda x:x["corr"])
    best = seed_data[-1]

    out_dir=os.path.join(base_dir,"filter_plot")
    os.makedirs(out_dir,exist_ok=True)

    roi_label=ROI_LABELS.get(roi,f"ROI{roi}")

    # kernel plot

    delays=np.linspace(-1,0,len(seed_data[0]["kernel"]))

    plot_kernel_overlay(
        seed_data,
        delays,
        os.path.join(out_dir,"linear_filter_kernel.pdf"),
        f"Linear filter {roi_label}"
    )

    # nonlinear plot

    plot_nonlinear(
        seed_data,
        os.path.join(out_dir,"nonlinear_g_vs_u.pdf"),
        f"Nonlinear {roi_label}",
        max_points
    )


    # best kernel

    plot_kernel_best(
        best,
        delays,
        os.path.join(out_dir,"best_linear_filter_kernel.pdf"),
        f"Best Linear filter {roi_label}"
    )

    # best nonlinear

    plot_nonlinear_best(
        best,
        os.path.join(out_dir,"best_nonlinear_g_vs_u.pdf"),
        f"Best Nonlinear {roi_label}",
        max_points
    )

# ==========================================================
# main
# ==========================================================

def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("root_dir")

    parser.add_argument("--max_points",type=int,default=80000)

    args=parser.parse_args()

    for roi in range(1,15):

        base_dir=os.path.join(args.root_dir,f"roi_{roi}","band_full")

        print("RUN",base_dir)

        run_for_roi(base_dir,[1.0],args.max_points,roi)


if __name__=="__main__":
    main()