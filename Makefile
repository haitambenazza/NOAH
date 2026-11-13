build:
	sudo chmod +x ./setup.sh
	./setup.sh
	pip3 install -r requirements.txt
	pip3 install --no-deps "openwakeword>=0.6.0"
	python3 -c "import openwakeword.utils; openwakeword.utils.download_models()"

run:
	python3 main.py