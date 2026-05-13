# src/tools/k_aVSk_f.py
"""
ROIごとの
kfi/ka と kfr/ka の距離マトリックスを作るツール。
実行例:
uv run python src/tools/k_aVSk_f.py scripts/spring04/scripts/limit
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROI_LABELS = {
1:"CBC1",2:"CBC2",3:"CBC3a",4:"CBC3b",5:"CBC4",
6:"CBC5t",7:"CBC5o",8:"CBC5i",9:"CBCX",10:"CBC6",
11:"CBC7",12:"CBC8",13:"CBC9",14:"RBC"
}

def read_param(path):

    if not os.path.exists(path):
        return None

    try:
        return float(np.genfromtxt(path))
    except:
        return None


def read_corr(seed_dir):

    path=os.path.join(seed_dir,"correlation.txt")

    if not os.path.exists(path):
        return None

    try:
        return float(np.genfromtxt(path))
    except:
        return None


def get_best_params(base_dir):

    best_corr=None
    best_params=None

    seeds=sorted(glob.glob(os.path.join(base_dir,"seed_*")))

    for s in seeds:

        ka=read_param(os.path.join(s,"ka.txt"))
        kfi=read_param(os.path.join(s,"kfi.txt"))
        kfr=read_param(os.path.join(s,"kfr.txt"))

        if None in (ka,kfi,kfr):
            continue

        corr=read_corr(s)

        if corr is None:
            continue

        if best_corr is None or corr>best_corr:

            best_corr=corr
            best_params=(ka,kfi,kfr)

    return best_params


def build_matrix(kfi,kfr,ka):

    x=kfi/ka
    y=kfr/ka

    N=len(x)

    matrix=np.zeros((N,N))

    for i in range(N):
        for j in range(N):

            dx=x[i]-x[j]
            dy=y[i]-y[j]

            matrix[i,j]=np.sqrt(dx**2+dy**2)

    return matrix


def save_outputs(matrix,labels,out_dir):

    os.makedirs(out_dir,exist_ok=True)

    df=pd.DataFrame(matrix,index=labels,columns=labels)

    # matrix table

    fig,ax=plt.subplots(figsize=(10,10))
    ax.axis("off")

    table=ax.table(
        cellText=np.round(df.values,3),
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center"
    )

    table.scale(1.2,1.2)

    plt.title("Kinetics Distance Matrix")

    fig.savefig(os.path.join(out_dir,"kinetics_matrix.pdf"),
                bbox_inches="tight")

    plt.close()

    # heatmap

    fig=plt.figure(figsize=(7,6))

    plt.imshow(matrix,cmap="viridis")

    plt.colorbar(label="distance")

    plt.xticks(range(len(labels)),labels,rotation=45)
    plt.yticks(range(len(labels)),labels)

    plt.title("Kinetics similarity between bipolar subtypes")

    plt.tight_layout()

    fig.savefig(os.path.join(out_dir,"kinetics_heatmap.pdf"),
                bbox_inches="tight")

    plt.close()


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("root_dir")
    parser.add_argument("--objective",default="band_full")

    args=parser.parse_args()

    ka=[]
    kfi=[]
    kfr=[]
    labels=[]

    for roi in range(1,15):

        base=os.path.join(args.root_dir,f"roi_{roi}",args.objective)

        print("loading",base)

        p=get_best_params(base)

        if p is None:
            print("skip roi",roi)
            continue

        a,fi,fr=p

        ka.append(a)
        kfi.append(fi)
        kfr.append(fr)

        labels.append(ROI_LABELS[roi])

    ka=np.array(ka)
    kfi=np.array(kfi)
    kfr=np.array(kfr)

    matrix=build_matrix(kfi,kfr,ka)

    save_outputs(matrix,labels,
                 os.path.join(args.root_dir,"kinetics_map"))

    print("=== DONE ===")


if __name__=="__main__":
    main()