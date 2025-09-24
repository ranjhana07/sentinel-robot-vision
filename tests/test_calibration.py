import cv2
import numpy as np
import pytest

from sentinel.export.calibration import ImageFolderCalibrationReader, list_images


def write_image(path, height=120, width=160, value=128):
    cv2.imwrite(str(path), np.full((height, width, 3), value, dtype=np.uint8))


@pytest.fixture
def image_folder(tmp_path):
    for index in range(5):
        write_image(tmp_path / f"face_{index}.jpg", value=index * 40)
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    return tmp_path


def test_list_images_filters_and_sorts(image_folder):
    names = [path.name for path in list_images(image_folder)]
    assert names == [f"face_{index}.jpg" for index in range(5)]


def test_reader_yields_batches_until_exhausted(image_folder):
    reader = ImageFolderCalibrationReader(image_folder, "images", input_size=64)
    batches = []
    while (batch := reader.get_next()) is not None:
        batches.append(batch)
    assert len(batches) == 5
    assert set(batches[0]) == {"images"}
    assert batches[0]["images"].shape == (1, 3, 64, 64)
    assert batches[0]["images"].dtype == np.float32


def test_reader_respects_max_images(image_folder):
    reader = ImageFolderCalibrationReader(image_folder, "images", input_size=32, max_images=2)
    assert len(reader) == 2


def test_rewind_restarts_iteration(image_folder):
    reader = ImageFolderCalibrationReader(image_folder, "images", input_size=32)
    first = reader.get_next()
    reader.get_next()
    reader.rewind()
    again = reader.get_next()
    assert np.array_equal(first["images"], again["images"])


def test_unreadable_files_are_skipped(image_folder):
    (image_folder / "broken.png").write_bytes(b"not an image")
    reader = ImageFolderCalibrationReader(image_folder, "images", input_size=32)
    count = 0
    while reader.get_next() is not None:
        count += 1
    assert count == 5


def test_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        ImageFolderCalibrationReader(tmp_path, "images")
