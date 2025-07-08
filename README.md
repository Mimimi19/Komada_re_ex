# Komada_re_ex

ローカルディレクトリで動かす場合は
docker build -t baccus_model .
docker run -v "$(pwd)/results:/app/results" baccus_model

ワークステーションで動かす場合は
mkdir Re_experiment
cd Re_experiment
git init
git remote -v
git remote add upstream /home/coder/Re_experiment/src/BaccusModel.py
git fetch upstream
git checkout main
cd src
python3 -m venv venv
pip install tqdm numpy scipy pyyaml matplotlib
source venv/bin/activate
python3 /home/coder/Re_experiment/src/BaccusModel.py
