PYTHON ?= python
CONFIG ?= configs/default.yaml

.PHONY: install dev test export quantize bench run clean

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

export:
	sentinel export models/face-yolov8n.pt --output models/face-yolov8n.onnx

quantize:
	sentinel quantize models/face-yolov8n.onnx --mode static --calibration-dir data/calibration --output models/face-yolov8n-int8.onnx

bench:
	sentinel bench models/face-yolov8n.onnx models/face-yolov8n-int8.onnx --report-dir reports

run:
	sentinel run --config $(CONFIG)

clean:
	rm -rf build dist .pytest_cache reports
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
