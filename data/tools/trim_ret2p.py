# data/tools/trim_ret2p.py
import numpy as np

# 抽出したい観測対象（0始まり）
target_indices = [1, 5, 11, 2]  # 観測対象のリスト
# 入力ファイル
input_file = "data/ret2p/response_data_repeat_1.txt"
# データ読み込み
data = np.loadtxt(input_file)
for target_index in target_indices:
    # 出力ファイル
    output_file = f"data/ret2p/trim/roi_{target_index}.txt"

    # shape確認（安全対策）
    assert data.shape[1] > target_index, "指定した観測対象が範囲外です"

    # 列を抽出
    extracted = data[:, target_index]

    # 保存（1列）
    np.savetxt(output_file, extracted, fmt="%.6f")

    print(f"抽出完了: shape = {extracted.shape}")
