# sentinel-robot-vision

Face detection with spoken responses for small surveillance robots, plus a log summarizer for the events the robot writes.

The detector runs on ONNX Runtime with its own preprocessing, decoding and NMS, so the robot does not need ultralytics or torch at runtime. Ultralytics is only used on the training machine to train and export the model.

## Architecture

```
  camera / video / picamera
            |
     ThreadedCapture  (bounded queue, drops stale frames)
            |
   letterbox -> ONNX Runtime -> decode -> NMS
            |
        IouTracker  (greedy IoU, optional velocity prediction)
            |
      IdentityResolver  (unknown by default, gallery hook)
            |
       SpeechPolicy  (cooldowns, zones, quiet hours, rate cap)
            |                              |
      SpeechWorker (priority queue)    RobotLogger (rotating JSONL)
            |                              |
   pyttsx3 / piper backend                 v
                                   logs ingest -> chunks -> ChromaDB
                                                      |
                                    logs summarize -> LangChain -> LLM
```

The capture thread and the speech worker are separate threads, so a slow audio device or a slow camera read never blocks inference.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tts]"
cp .env.example .env
```

Extras: `train` (ultralytics), `rag` (ChromaDB, sentence-transformers, LangChain), `pi` (picamera2), `dev` (pytest). On a Raspberry Pi install `picamera2` from apt rather than pip.

A Dockerfile for arm64 and a systemd unit (`deploy/sentinel.service`) are included.

## Configuration

Runtime behaviour lives in `configs/default.yaml`: detector thresholds, source and reconnect policy, tracker, speech rules, snapshots, log rotation and RAG settings. Speech zones are rectangles in normalised frame coordinates:

```yaml
speech:
  zones:
    - name: restricted
      area: [0.55, 0.0, 1.0, 1.0]
      phrase: "This area is restricted. Please step back."
      applies_to: unknown
```

Any value can be overridden on the command line, `sentinel run --set detector.conf_threshold=0.5`, and `sentinel config show` prints the effective configuration. LLM settings come from environment variables, see `.env.example`:

| variable | default |
| --- | --- |
| `SENTINEL_LLM_PROVIDER` | `anthropic` |
| `SENTINEL_LLM_MODEL` | `claude-sonnet-5-5` |
| `SENTINEL_LLM_TEMPERATURE` | `0` |
| `ANTHROPIC_API_KEY` | required for `logs summarize` |

## Usage

```
sentinel train --data configs/face_dataset.yaml
sentinel export models/face_yolov8n.pt --output models/face-fp32.onnx --image-size 416
sentinel frames clip.mp4 --output-dir data/calibration
sentinel quantize models/face-fp32.onnx --mode static --calibration-dir data/calibration \
    --output models/face-int8-static.onnx
sentinel bench models/face-fp32.onnx models/face-int8-static.onnx --pytorch models/face_yolov8n.pt
sentinel compare models/face-fp32.onnx models/face-int8-static.onnx --images data/heldout
sentinel run --config configs/default.yaml --display
```

Logs:

```
sentinel logs ingest logs/ examples/sample_robot.jsonl
sentinel logs stats logs/robot.jsonl --since 6h
sentinel logs summarize --since 2025-11-04T09:00:00Z --until 2025-11-04T12:00:00Z
sentinel logs summarize --since 6h --question "Were there any camera failures?"
```

Ingest parses JSONL or plain `timestamp LEVEL message` lines, groups them into time windows, embeds each window with sentence-transformers and stores it in ChromaDB. A summary retrieves the windows closest to the question inside the requested range and passes them to the LLM through LangChain.

## Results

Everything below was produced on one machine by the scripts in this repository, and the raw files are in `results/`.

Hardware and software: Intel Core i5-13420H (12 logical cores, 4 threads used for inference), Windows 11, Python 3.12.10, onnxruntime 1.32.0, torch 2.15.0 (CPU).

### Model and data

- Weights: `face_yolov8n.pt` from the Bingsu/adetailer repository on Hugging Face, a YOLOv8n face detector trained by a third party. It was not trained here.
- Input size 416. Exported with `sentinel export` at opset 13.
- Images: the 64 photos in the `tests/unit/dataset` folder of the deepface repository. Sorted by name, odd positions (32 images) calibrate the static quantizer and even positions (32 images) are held out for the agreement check.

### Latency

Command (`scripts/run_benchmarks.sh`, with the paths above set through its environment variables):

```
sentinel bench face-fp32.onnx face-int8-dynamic.onnx face-int8-static.onnx \
    --pytorch face_yolov8n.pt --image-size 416 --threads 4 --runs 200
```

Raw results: `results/benchmark.csv`, `results/benchmark.json`.

| model | size (MB) | mean (ms) | p50 (ms) | p95 (ms) | FPS (mean) | FPS (median) | speedup (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 (ultralytics network, no NMS) | 6.2 | 805.8 | 507.4 | 2676.2 | 1.2 | 2.0 | 1.00x |
| ONNX Runtime FP32 | 12.2 | 100.0 | 89.0 | 247.2 | 10.0 | 11.2 | 5.70x |
| ONNX Runtime INT8 dynamic | 3.3 | 444.6 | 395.5 | 717.3 | 2.2 | 2.5 | 1.28x |
| ONNX Runtime INT8 static | 3.5 | 22.3 | 20.6 | 32.5 | 44.9 | 48.7 | 24.68x |

These are single forward passes on random input, without preprocessing, NMS or tracking. The laptop CPU has performance and efficiency cores and the PyTorch row in particular was noisy (mean 806 ms against a median of 507 ms and a p95 of 2.7 s), so read its median with caution. Dynamic INT8 came out slower than FP32 on this CPU, so only static QDQ quantization is worth shipping here.

### INT8 agreement with FP32

`sentinel compare` runs the FP32 and INT8 models on the 32 held-out images (40 faces found by the FP32 model at confidence 0.35) and matches boxes at IoU 0.5. Raw results: `results/agreement_static.json`, `results/agreement_dynamic.json`.

| model | recall | precision | mean IoU of matches | mean abs. score difference |
| --- | ---: | ---: | ---: | ---: |
| INT8 static | 1.00 | 1.00 | 0.982 | 0.019 |
| INT8 dynamic | 1.00 | 1.00 | 0.982 | 0.010 |

This measures agreement with the FP32 model, not accuracy against ground-truth labels. Keeping the YOLOv8 detection head in FP32 matters: with the whole graph quantized, the box coordinates and class scores share one output tensor and one scale, and the static INT8 model found no faces at all on the same images. `sentinel quantize` therefore excludes the head by default.

### Log retrieval

`python scripts/evaluate_logs.py --mode retrieval` ingests `examples/sample_robot.jsonl` (327 lines of a synthetic robot day with injected incidents, written by `scripts/generate_sample_logs.py`) and checks, for each of the 8 questions in `eval/dataset.jsonl`, whether the retrieved windows contain the log events the answer depends on. Embeddings are `all-MiniLM-L6-v2` and the retriever returns the top 6 windows of 10 minutes. Raw results: `results/retrieval_eval.json`.

| metric | value |
| --- | ---: |
| questions | 8 |
| all required events retrieved | 7 of 8 (0.875) |
| mean coverage of required events | 0.9375 |

The miss is the question that asks for every error of the whole day, which needs more windows than the top 6.

### Not measured here

- Frame rate on a Raspberry Pi. No Pi was available. To reproduce, copy the repository and the three ONNX files to the Pi, install with `pip install -e ".[tts]"`, and run the same `sentinel bench` command with `--threads 4`, then `sentinel run --set speech.enabled=false --max-frames 500` to see the end to end rate in the `fps` events of `logs/robot.jsonl`.
- LLM summary quality and the LangSmith judge scores. They need an `ANTHROPIC_API_KEY` and a LangSmith key. `python scripts/evaluate_logs.py --mode langsmith` uploads the dataset, runs the summarizer for each question and records an LLM-as-judge correctness score and the retrieval coverage as a LangSmith experiment.
- Time saved reviewing logs. This needs real operators and real logs. A fair comparison is to time a reviewer finding the incidents of one day from raw files against reading `sentinel logs summarize` output for the same day.

## Project layout

```
configs/            default runtime config, training dataset definition
deploy/             systemd unit
eval/               question and reference pairs for the summarizer
examples/           synthetic sample robot log
results/            raw benchmark, agreement and retrieval results
scripts/            benchmark runner, log generator, evaluation script
src/sentinel/
  capture/          sources, picamera, threaded capture, reconnect
  detection/        letterbox, decode, nms, onnx engine, training wrapper
  export/           onnx export, quantization, calibration, agreement
  rag/              parsing, chunking, store, summarizer, evaluation, stats
  robotlog/         rotating jsonl logger
  speech/           policy, cooldowns, rate limit, queue, tts backends
  tracking/         iou tracker, identity resolvers
  pipeline.py       wiring for `sentinel run`
  cli.py            typer entry point
tests/              unit tests
```

## Tests

```
pip install -e ".[dev]" onnx
python -m pytest
```

The tests cover pure logic and small synthetic ONNX graphs and need no model weights, camera or network.

## Docker

```
docker buildx build --platform linux/arm64 -t sentinel-robot-vision .
docker run --rm --device /dev/snd --device /dev/video0 \
    -v $PWD/models:/app/models -v $PWD/logs:/app/logs sentinel-robot-vision
```

Add the log tooling with `--build-arg EXTRAS=tts,rag`. Unknown faces can be saved for review with `--set snapshots.enabled=true`, which writes cropped JPEGs to `data/snapshots` and keeps the newest 200.
