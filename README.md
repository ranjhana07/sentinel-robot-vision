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

## Log summaries

Robot events go to `logs/robot.jsonl` as one JSON object per line, rotated by size. To search and summarize them:

```
sentinel logs ingest logs/ examples/sample_robot.jsonl
sentinel logs summarize --since 2025-11-04T09:00:00Z --until 2025-11-04T12:00:00Z
sentinel logs summarize --since 6h --question "Were there any camera failures?"
```

Ingest parses JSONL or plain `timestamp LEVEL message` lines, groups them into time windows, embeds each window with sentence-transformers and stores it in ChromaDB. Summaries retrieve the windows closest to the question inside the requested range and hand them to the LLM through LangChain. The model is chosen with `SENTINEL_LLM_PROVIDER` and `SENTINEL_LLM_MODEL`; the default is `claude-sonnet-5-5` through `langchain-anthropic`, which needs `ANTHROPIC_API_KEY`.
