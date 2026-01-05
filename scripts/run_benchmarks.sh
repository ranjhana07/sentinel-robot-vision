#!/bin/bash
set -euo pipefail

WEIGHTS=${WEIGHTS:-models/face_yolov8n.pt}
CALIBRATION_DIR=${CALIBRATION_DIR:-data/calibration}
HELDOUT_DIR=${HELDOUT_DIR:-data/heldout}
MODEL_DIR=${MODEL_DIR:-models}
RESULTS_DIR=${RESULTS_DIR:-results}
IMAGE_SIZE=${IMAGE_SIZE:-416}
THREADS=${THREADS:-4}
RUNS=${RUNS:-200}

mkdir -p "$MODEL_DIR" "$RESULTS_DIR"

sentinel export "$WEIGHTS" --output "$MODEL_DIR/face-fp32.onnx" --image-size "$IMAGE_SIZE"
sentinel quantize "$MODEL_DIR/face-fp32.onnx" --mode static --calibration-dir "$CALIBRATION_DIR" \
    --output "$MODEL_DIR/face-int8-static.onnx" --image-size "$IMAGE_SIZE"
sentinel quantize "$MODEL_DIR/face-fp32.onnx" --mode dynamic --output "$MODEL_DIR/face-int8-dynamic.onnx"

sentinel bench "$MODEL_DIR/face-fp32.onnx" "$MODEL_DIR/face-int8-dynamic.onnx" "$MODEL_DIR/face-int8-static.onnx" \
    --pytorch "$WEIGHTS" --image-size "$IMAGE_SIZE" --threads "$THREADS" --runs "$RUNS" \
    --report-dir "$RESULTS_DIR" --stem benchmark

sentinel compare "$MODEL_DIR/face-fp32.onnx" "$MODEL_DIR/face-int8-static.onnx" \
    --images "$HELDOUT_DIR" --image-size "$IMAGE_SIZE" --output "$RESULTS_DIR/agreement_static.json"
sentinel compare "$MODEL_DIR/face-fp32.onnx" "$MODEL_DIR/face-int8-dynamic.onnx" \
    --images "$HELDOUT_DIR" --image-size "$IMAGE_SIZE" --output "$RESULTS_DIR/agreement_dynamic.json"
