build:
	sudo chmod +x ./setup.sh
	./setup.sh
	pip3 install --no-deps "openwakeword>=0.6.0"
	pip3 install --upgrade faster-whisper
	python3 -c "import importlib.metadata as m, pathlib; p = pathlib.Path(m.distribution('openwakeword')._path) / 'METADATA'; p.write_text('\n'.join(l for l in p.read_text().splitlines() if 'tflite-runtime' not in l) + '\n')"
	python3 -c "import openwakeword.utils; openwakeword.utils.download_models()"
	pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python
	pip3 install opencv-python
	pip3 install -r requirements.txt
run:
	python3 main.py