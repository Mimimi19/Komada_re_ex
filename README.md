# Komada_re_ex

# ローカルディレクトリで動かす場合は

```
docker build -t baccus_model .
docker run -v "$(pwd)/results:/app/results" baccus_model
```

# ワークステーションで動かす場合は

```
cd ~
git clone -b 3-reset_result_location https://github.com/Mimimi19/Komada_re_ex.git Re_experiment
cd Re_experiment
python3 -m venv venv
source venv/bin/activate
pip install tqdm numpy scipy pyyaml matplotlib numba
cd src
python3 BaccusModel.py
```
