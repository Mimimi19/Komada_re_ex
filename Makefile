.PHONY: init

init:
	git clone https://github.com/Mimimi19/Komada_re_ex.git Re_experiment
	cd Re_experiment
	@echo ".envとca.crtの設定を行ってください。"

#探索するデータの選択
cb1:
	uv add -r requirements.txt
	uv run src/BaccusModel.py data=Ucb1

cb2:
	uv add -r requirements.txt
	uv run src/BaccusModel.py data=Ucb2

ret2p:
	uv add -r requirements.txt
	uv run src/BaccusModel.py data=ret2p-1

#ログの確認
log:
	cd Re_experiment/
	tail -f scripts/lnk_model_debug.log | nl

#MLflow
mlflow:
	cd Re_experiment/
	source venv/bin/activate
	cd scripts
	mlflow ui