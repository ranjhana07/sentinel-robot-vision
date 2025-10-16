# sentinel-robot-vision

Face detection with spoken responses for small surveillance robots, plus a log summarizer for the events the robot writes.

The detector runs on ONNX Runtime with no ultralytics dependency at runtime, so it installs on a Raspberry Pi without torch. Ultralytics is only needed on the training machine to train and export the model.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tts]"
cp .env.example .env
```

On a training machine add the `train` extra: `pip install -e ".[train]"`.

## Usage

Export trained weights to ONNX and quantize them:

```
sentinel export models/face-yolov8n.pt --output models/face-yolov8n.onnx
sentinel quantize models/face-yolov8n.onnx --mode static \
    --calibration-dir data/calibration --output models/face-yolov8n-int8.onnx
```

Compare the models:

```
sentinel bench models/face-yolov8n.onnx models/face-yolov8n-int8.onnx
```

Run the robot loop:

```
sentinel run --config configs/default.yaml --display
```

## Configuration

Everything about the detector, source, tracker, speech rules and log rotation lives in `configs/default.yaml`. LLM settings come from environment variables prefixed with `SENTINEL_`, see `.env.example`.

## Tests

```
pip install -e ".[dev]" onnx
python -m pytest
```

The tests cover pure logic only and need no model weights, camera or network.
