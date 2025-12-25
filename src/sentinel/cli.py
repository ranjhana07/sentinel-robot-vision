import json
import signal
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Sentinel robot vision toolkit")


class QuantMode(str, Enum):
    dynamic = "dynamic"
    static = "static"


@app.command()
def run(
    config: Annotated[Path, typer.Option(help="yaml config file")] = Path("configs/default.yaml"),
    source: Annotated[Optional[str], typer.Option(help="camera index or video path override")] = None,
    model: Annotated[Optional[Path], typer.Option(help="onnx model override")] = None,
    display: Annotated[bool, typer.Option(help="show an overlay window")] = False,
    max_frames: Annotated[Optional[int], typer.Option(help="stop after this many frames")] = None,
    overrides: Annotated[list[str], typer.Option("--set", help="override a config value, e.g. detector.conf_threshold=0.5")] = [],
) -> None:
    from sentinel.config import apply_overrides, load_config
    from sentinel.pipeline import build_pipeline

    assignments = list(overrides)
    if source is not None:
        assignments.append(f"source.kind={'camera' if source.isdigit() else 'video'}")
        assignments.append(f"source.target={source if source.isdigit() else json.dumps(source)}")
    if model is not None:
        assignments.append(f"detector.model_path={json.dumps(str(model))}")
    app_config = apply_overrides(load_config(config), assignments)
    pipeline = build_pipeline(app_config, display=display)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: pipeline.request_stop())
    frames = pipeline.run(max_frames=max_frames)
    typer.echo(f"processed {frames} frames")


@app.command()
def export(
    weights: Annotated[Path, typer.Argument(help="trained .pt weights")],
    output: Annotated[Optional[Path], typer.Option(help="destination .onnx path")] = None,
    image_size: Annotated[int, typer.Option(help="square input size")] = 416,
    opset: Annotated[int, typer.Option(help="13 or higher, needed for per-channel int8")] = 13,
) -> None:
    from sentinel.export.onnx_export import export_onnx

    path = export_onnx(weights, output, image_size=image_size, opset=opset)
    typer.echo(f"wrote {path}")


@app.command()
def quantize(
    model: Annotated[Path, typer.Argument(help="fp32 .onnx model")],
    output: Annotated[Path, typer.Option(help="destination .onnx path")],
    mode: Annotated[QuantMode, typer.Option()] = QuantMode.static,
    calibration_dir: Annotated[Optional[Path], typer.Option(help="images for static mode")] = None,
    image_size: Annotated[int, typer.Option()] = 416,
    max_images: Annotated[int, typer.Option()] = 200,
    method: Annotated[str, typer.Option(help="minmax, entropy or percentile")] = "minmax",
    per_channel: Annotated[bool, typer.Option(help="per-channel weight scales")] = True,
    preprocess: Annotated[bool, typer.Option(help="run onnxruntime shape inference first")] = True,
    exclude_head: Annotated[bool, typer.Option(help="keep the detection head in fp32")] = True,
) -> None:
    from sentinel.export.quantize import CALIBRATION_METHODS, quantize_dynamic_int8, quantize_static_int8

    if method not in CALIBRATION_METHODS:
        raise typer.BadParameter(f"method must be one of {sorted(CALIBRATION_METHODS)}")
    if mode is QuantMode.dynamic:
        path = quantize_dynamic_int8(model, output)
    else:
        if calibration_dir is None:
            raise typer.BadParameter("static mode needs --calibration-dir")
        path = quantize_static_int8(
            model, output, calibration_dir, image_size, max_images, method, per_channel, preprocess, exclude_head
        )
    typer.echo(f"wrote {path}")


@app.command()
def bench(
    models: Annotated[list[Path], typer.Argument(help="onnx models to compare, baseline first")],
    pytorch: Annotated[Optional[Path], typer.Option(help=".pt weights to add as the first row")] = None,
    runs: Annotated[int, typer.Option()] = 100,
    warmup: Annotated[int, typer.Option()] = 10,
    image_size: Annotated[int, typer.Option()] = 416,
    threads: Annotated[int, typer.Option()] = 4,
    report_dir: Annotated[Path, typer.Option()] = Path("reports"),
    stem: Annotated[str, typer.Option(help="report file name without extension")] = "benchmark",
) -> None:
    from sentinel.bench import benchmark_model, benchmark_pytorch, render_markdown, write_reports

    results = []
    if pytorch is not None:
        results.append(benchmark_pytorch(pytorch, image_size, runs, warmup, threads))
    results.extend(benchmark_model(path, image_size, runs, warmup, threads) for path in models)
    typer.echo(render_markdown(results))
    settings = {"image_size": image_size, "runs": runs, "warmup": warmup, "threads": threads}
    paths = write_reports(results, report_dir, settings, stem)
    typer.echo("wrote " + ", ".join(str(path) for path in paths))


@app.command()
def compare(
    reference: Annotated[Path, typer.Argument(help="reference (fp32) onnx model")],
    candidate: Annotated[Path, typer.Argument(help="model to check, for example the int8 one")],
    images: Annotated[Path, typer.Option(help="folder of test images")],
    image_size: Annotated[int, typer.Option()] = 416,
    conf: Annotated[float, typer.Option()] = 0.35,
    limit: Annotated[int, typer.Option()] = 200,
    output: Annotated[Optional[Path], typer.Option(help="write the summary as json")] = None,
) -> None:
    import json
    from dataclasses import asdict

    import cv2

    from sentinel.detection.onnx_engine import OnnxFaceDetector
    from sentinel.export.agreement import compare_detectors
    from sentinel.export.calibration import list_images

    frames_list = [cv2.imread(str(path)) for path in list_images(images, limit)]
    frames_list = [frame for frame in frames_list if frame is not None]
    summary = compare_detectors(
        OnnxFaceDetector(reference, image_size, conf),
        OnnxFaceDetector(candidate, image_size, conf),
        frames_list,
    )
    text = json.dumps(asdict(summary), indent=2)
    typer.echo(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


@app.command()
def frames(
    video: Annotated[Path, typer.Argument(help="video to sample")],
    output_dir: Annotated[Path, typer.Option()] = Path("data/calibration"),
    every: Annotated[int, typer.Option(help="keep every Nth frame")] = 15,
    limit: Annotated[int, typer.Option()] = 200,
) -> None:
    from sentinel.export.frames import extract_frames

    written = extract_frames(video, output_dir, every=every, limit=limit)
    typer.echo(f"wrote {len(written)} frames to {output_dir}")


@app.command()
def train(
    data: Annotated[Path, typer.Option()] = Path("configs/face_dataset.yaml"),
    base_weights: Annotated[str, typer.Option()] = "yolov8n.pt",
    epochs: Annotated[int, typer.Option()] = 100,
    image_size: Annotated[int, typer.Option()] = 640,
    batch: Annotated[int, typer.Option()] = 16,
    device: Annotated[Optional[str], typer.Option()] = None,
) -> None:
    from sentinel.detection.train import train_face_detector

    best = train_face_detector(data, base_weights, epochs, image_size, batch, device)
    typer.echo(f"best weights: {best}")


config_app = typer.Typer(no_args_is_help=True, help="Inspect configuration")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    config: Annotated[Path, typer.Option()] = Path("configs/default.yaml"),
    overrides: Annotated[list[str], typer.Option("--set", help="apply an override before printing")] = [],
) -> None:
    import yaml

    from sentinel.config import apply_overrides, load_config

    effective = apply_overrides(load_config(config), list(overrides))
    typer.echo(yaml.safe_dump(effective.model_dump(mode="json"), sort_keys=False))


@config_app.command("check")
def config_check(
    config: Annotated[Path, typer.Option()] = Path("configs/default.yaml"),
) -> None:
    from pydantic import ValidationError

    from sentinel.config import load_config

    try:
        loaded = load_config(config)
    except ValidationError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1)
    model = loaded.detector.model_path
    typer.echo(f"{config} is valid")
    if not model.exists():
        typer.echo(f"note: model file {model} does not exist yet")


logs_app = typer.Typer(no_args_is_help=True, help="Ingest and summarize robot logs")
app.add_typer(logs_app, name="logs")


@logs_app.command("ingest")
def logs_ingest(
    paths: Annotated[list[Path], typer.Argument(help="log files or directories (jsonl or text)")],
    config: Annotated[Path, typer.Option()] = Path("configs/default.yaml"),
    window_minutes: Annotated[Optional[int], typer.Option(help="chunk window override")] = None,
) -> None:
    from sentinel.config import load_config
    from sentinel.rag.ingest import build_chunks
    from sentinel.rag.store import LogStore

    rag = load_config(config).rag
    chunks = build_chunks(paths, window_minutes or rag.window_minutes, rag.max_chunk_chars)
    store = LogStore.from_config(rag)
    stored = store.add_chunks(chunks)
    typer.echo(f"stored {stored} chunks, collection now holds {store.count()}")


@logs_app.command("summarize")
def logs_summarize(
    since: Annotated[str, typer.Option(help="ISO time or relative value such as 2h")],
    until: Annotated[str, typer.Option()] = "now",
    question: Annotated[Optional[str], typer.Option(help="focus the summary on a question")] = None,
    config: Annotated[Path, typer.Option()] = Path("configs/default.yaml"),
) -> None:
    from sentinel.config import Settings, load_config
    from sentinel.rag.store import LogStore
    from sentinel.rag.summarize import LogSummarizer, build_llm
    from sentinel.rag.timeparse import parse_time_argument

    start = parse_time_argument(since)
    end = parse_time_argument(until)
    if start >= end:
        raise typer.BadParameter("--since must be earlier than --until")
    rag = load_config(config).rag
    summarizer = LogSummarizer(LogStore.from_config(rag), build_llm(Settings()), top_k=rag.top_k)
    summary = summarizer.summarize(start, end, question)
    typer.echo(summary.text)
    typer.echo(f"\nchunks used: {len(summary.chunk_ids)}")


if __name__ == "__main__":
    app()
