build:
	./setup.sh
	pip3 install -r requirements.txt
	pip3 install --no-deps "openwakeword>=0.6.0"

run:
	python3 main.py