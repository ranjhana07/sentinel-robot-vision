| model | size (MB) | mean (ms) | p50 (ms) | p95 (ms) | FPS (mean) | FPS (median) | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| face_yolov8n.pt (pytorch) | 6.2 | 805.8 | 507.4 | 2676.2 | 1.2 | 2.0 | 1.00x |
| face-fp32.onnx | 12.2 | 100.0 | 89.0 | 247.2 | 10.0 | 11.2 | 5.70x |
| face-int8-dynamic.onnx | 3.3 | 444.6 | 395.5 | 717.3 | 2.2 | 2.5 | 1.28x |
| face-int8-static.onnx | 3.5 | 22.3 | 20.6 | 32.5 | 44.9 | 48.7 | 24.68x |
