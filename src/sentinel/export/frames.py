from pathlib import Path

import cv2


def extract_frames(
    video_path: Path | str,
    output_dir: Path | str,
    every: int = 15,
    limit: int | None = 200,
) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index = 0
    try:
        while limit is None or len(written) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every == 0:
                path = destination / f"frame_{index:06d}.jpg"
                cv2.imwrite(str(path), frame)
                written.append(path)
            index += 1
    finally:
        capture.release()
    return written
