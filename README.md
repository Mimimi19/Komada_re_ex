# Komada_re_ex

# ローカルディレクトリで動かす場合は
```
docker build -t baccus_model .
docker run -v "$(pwd)/results:/app/results" baccus_model
```
# ワークステーションで動かす場合は
```
cd ~
mkdir Re_experiment
cd Re_experiment  
git clone https://github.com/Mimimi19/Komada_re_ex.git Komada_re_ex
cd Komada_re_ex
python3 -m venv venv
source venv/bin/activate
pip install tqdm numpy scipy pyyaml matplotlib
cd src
python3 BaccusModel.py
```
