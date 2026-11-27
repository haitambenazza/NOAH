VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: build venv run

venv:
	python3 -m venv --system-site-packages $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install --no-deps "openwakeword>=0.6.0"
	$(PYTHON) -c "import importlib.metadata as m, pathlib; p = pathlib.Path(m.distribution('openwakeword')._path) / 'METADATA'; p.write_text('\n'.join(l for l in p.read_text().splitlines() if 'tflite-runtime' not in l and 'ai-edge-litert' not in l) + '\n')"
	$(PYTHON) -c "import openwakeword.utils; openwakeword.utils.download_models()"

build:
	sudo chmod +x ./setup.sh
	./setup.sh
	$(MAKE) venv

run:
	@if [ -x "$(PYTHON)" ]; then $(PYTHON) main.py; else python3 main.py; fi
