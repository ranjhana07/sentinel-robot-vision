from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Sentinel robot vision toolkit")


class QuantMode(str, Enum):
    dynamic = "dynamic"
    static = "static"


@app.command()
def export(
    weights: Annotated[Path, typer.Argument(help="trained .pt weights")],
    output: Annotated[Optional[Path], typer.Option(help="destination .onnx path")] = None,
    image_size: Annotated[int, typer.Option(help="square input size")] = 416,
    opset: Annotated[int, typer.Option()] = 12,
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
) -> None:
    from sentinel.export.quantize import quantize_dynamic_int8, quantize_static_int8

    if mode is QuantMode.dynamic:
        path = quantize_dynamic_int8(model, output)
    else:
        if calibration_dir is None:
            raise typer.BadParameter("static mode needs --calibration-dir")
        path = quantize_static_int8(model, output, calibration_dir, image_size, max_images)
    typer.echo(f"wrote {path}")


@app.command()
def bench(
    models: Annotated[list[Path], typer.Argument(help="models to compare, baseline first")],
    runs: Annotated[int, typer.Option()] = 100,
    warmup: Annotated[int, typer.Option()] = 10,
    image_size: Annotated[int, typer.Option()] = 416,
    threads: Annotated[int, typer.Option()] = 4,
    report_dir: Annotated[Path, typer.Option()] = Path("reports"),
) -> None:
    from sentinel.bench import benchmark_model, render_markdown, write_reports

    results = [benchmark_model(path, image_size, runs, warmup, threads) for path in models]
    typer.echo(render_markdown(results))
    markdown_path, csv_path = write_reports(results, report_dir)
    typer.echo(f"wrote {markdown_path} and {csv_path}")


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


if __name__ == "__main__":
    app()
