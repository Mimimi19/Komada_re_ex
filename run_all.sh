#!/bin/bash

# chmod +x run_all.sh
# ./run_all.sh

# エラーが出たらそこで止まるようにする（必要なければ set -e を削除）
set -e

echo "Starting Batch 1: Ucb1 (10s segments)"
uv run python main.py data=Ucb1 segment=10 hyper_params.objective_type=spearman

echo "Starting Batch 2: Ucb2 (10s segments)"
uv run python main.py data=Ucb2 segment=10 hyper_params.objective_type=spearman

echo "Starting Batch 3: ret2p-1 (10s segments)"
uv run python main.py data=ret2p-1 segment=10 hyper_params.objective_type=spearman

echo "Starting Batch 4: Ucb1 (5s segments)"
uv run python main.py data=Ucb1 segment=5 hyper_params.objective_type=spearman

echo "Starting Batch 5: Ucb2 (5s segments)"
uv run python main.py data=Ucb2 segment=5 hyper_params.objective_type=spearman

echo "Starting Batch 6: ret2p-1 (5s segments)"
uv run python main.py data=ret2p-1 segment=5 hyper_params.objective_type=spearman

echo "All batch jobs completed!"