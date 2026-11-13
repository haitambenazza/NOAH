build:
	sudo chmod +x ./setup.sh
	./setup.sh
	pip3 install --no-deps "openwakeword>=0.6.0"
	python3 -c "import importlib.metadata as m, pathlib; p = pathlib.Path(m.distribution('openwakeword')._path) / 'METADATA'; p.write_text('\n'.join(l for l in p.read_text().splitlines() if 'tflite-runtime' not in l) + '\n')"
	pip3 install -r requirements.txt
	python3 -c "import openwakeword.utils; openwakeword.utils.download_models()"
	pip3 install --upgrade faster-whisper

run:
	python3 main.py