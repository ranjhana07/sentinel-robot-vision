import shutil
from pathlib import Path


def export_onnx(
    weights: Path | str,
    output: Path | str | None = None,
    image_size: int = 416,
    opset: int = 13,
    simplify: bool = True,
) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = Path(
        model.export(format="onnx", imgsz=image_size, opset=opset, simplify=simplify, dynamic=False)
    )
    if output is None:
        return exported
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), destination)
    return destination
