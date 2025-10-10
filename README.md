# Komada_re_ex

# ローカルディレクトリで動かす場合

```
docker build -t baccus_model .
docker run -v "$(pwd)/results:/app/results" baccus_model
```

# ワークステーションで動かす場合

```
#cb1の時
cd ~
git clone https://github.com/Mimimi19/Komada_re_ex.git Re_experiment
cd Re_experiment
python3 -m venv venv
source venv/bin/activate
pip install tqdm numpy scipy pyyaml matplotlib numba hydra-core omegaconf antlr4-python3-runtime mlflow
python3 src/BaccusModel.py
#cb2のとき
cd ~
git clone https://github.com/Mimimi19/Komada_re_ex.git Re_experiment
cd Re_experiment
python3 -m venv venv
source venv/bin/activate
pip install tqdm numpy scipy pyyaml matplotlib numba hydra-core omegaconf antlr4-python3-runtime mlflow
python3 src/BaccusModel.py data=cb2

#別のターミナルを開いてログを表示
tail -f scripts/lnk_model_debug.log | nl

```

<!-- シミュレーションが終わらなくて、最新の学習データで再現したモデルなのですが


パラメータの定義いきをパラメータの動ける箇所をあぶり出し、足りないパラメータに割り当てる。
目的関数がG分布に従わないようのスピアマン相関係数だから計算しなせ。
\alpha を減らすこともできる、線形モデルの形から参照する過去がどの程度がをあぶり出し必要数のjを変更することで実現、また、リニアモデルのグラフは離散型地やから。分布図に直せ、ノンリニアもそう。 -->
