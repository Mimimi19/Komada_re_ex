import numpy as np # math.exp の代わりに np.exp を使用

def main(x_input, a, b1, b2):
    """
    非線形モデルの計算を行います。
    x_input: 入力値（スカラーまたはNumPy配列）
    a, b1, b2: 非線形パラメータ

    x_inputがNumPy配列の場合、要素ごとに計算が適用されます（ベクトル化）。
    """
    # np.exp はスカラーとNumPy配列の両方に対応し、要素ごとの計算を行う
    return a / (1 + b1 * np.exp(-b2 * x_input))

if __name__ == "__main__":
    # テスト用のコード (必要であれば追加)
    test_scalar = 0.5
    test_array = np.array([-1.0, 0.0, 1.0, 2.0])
    test_a, test_b1, test_b2 = 1.0, 2.0, 3.0

    print(f"Scalar result: {main(test_scalar, test_a, test_b1, test_b2)}")
    print(f"Array result: {main(test_array, test_a, test_b1, test_b2)}")