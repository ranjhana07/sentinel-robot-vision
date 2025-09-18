from pathlib import Path


def train_face_detector(
    data: Path | str,
    base_weights: str = "yolov8n.pt",
    epochs: int = 100,
    image_size: int = 640,
    batch: int = 16,
    device: str | None = None,
    project: Path | str = "runs",
    name: str = "face",
) -> Path:
    from ultralytics import YOLO

    model = YOLO(base_weights)
    results = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    return Path(results.save_dir) / "weights" / "best.pt"
