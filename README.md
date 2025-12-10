# Komada_re_ex

# ローカルディレクトリで動かす場合

```
docker build -t baccus_model .
docker run -v "$(pwd)/results:/app/results" baccus_model
```

# ワークステーションで動かす場合

```
git clone https://github.com/Mimimi19/Komada_re_ex.git Re_experiment
cd Re_experiment

.envとca.crtの設定

#cb1の時
uv add -r requirements.txt
uv run src/BaccusModel.py data=Ucb1

#cb2のとき
uv add -r requirements.txt
uv run src/BaccusModel.py data=Ucb2

#ret2pのとき
uv add -r requirements.txt
uv run src/BaccusModel.py data=ret2p-1

#I2を使わない時
hyper_params.use_I2=false

#別のターミナルを開いてログを表示
cd Re_experiment/
tail -f scripts/lnk_model_debug.log | nl

#MLflow
cd Re_experiment/
source venv/bin/activate
cd scripts
mlflow ui

DE法の戦略を変える場合
DE/rand/1/bin がデフォルト

optimization.strategy.mutation=
rand  best  randtobest  currenttobest

optimization.strategy.n_vectors=1  or  2

optimization.strategy.crossover=bin  or  exp
例：DE/best/2/bin　data=Ucb2
uv run src/BaccusModel.py data=Ucb2 optimization.strategy.mutation=best optimization.strategy.n_vectors=2 optimization.strategy.crossover=bin
```

<!-- シミュレーションが終わらなくて、最新の学習データで再現したモデルなのですが


パラメータの定義いきをパラメータの動ける箇所をあぶり出し、足りないパラメータに割り当てる。
目的関数がG分布に従わないようのスピアマン相関係数だから計算しなせ。
\alpha を減らすこともできる、線形モデルの形から参照する過去がどの程度がをあぶり出し必要数のjを変更することで実現、また、リニアモデルのグラフは離散型地やから。分布図に直せ、ノンリニアもそう。 -->
