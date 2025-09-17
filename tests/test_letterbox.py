import numpy as np
import pytest

from sentinel.detection.preprocess import compute_letterbox, letterbox, prepare_input


@pytest.mark.parametrize(
    "height,width,size,expected",
    [
        (480, 640, 416, (0.65, 416, 312, 0, 52)),
        (640, 480, 416, (0.65, 312, 416, 52, 0)),
        (416, 416, 416, (1.0, 416, 416, 0, 0)),
        (100, 200, 400, (2.0, 400, 200, 0, 100)),
    ],
)
def test_compute_letterbox(height, width, size, expected):
    scale, new_w, new_h, left, top = compute_letterbox(height, width, size)
    assert scale == pytest.approx(expected[0])
    assert (new_w, new_h, left, top) == expected[1:]


def test_letterbox_output_shape_and_padding():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    canvas, meta = letterbox(image, 416)
    assert canvas.shape == (416, 416, 3)
    assert meta.pad_top == 52
    assert canvas[0, 0].tolist() == [114, 114, 114]
    assert canvas[208, 208].tolist() == [255, 255, 255]


def test_prepare_input_is_normalised_nchw():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    tensor, meta = prepare_input(image, 320)
    assert tensor.shape == (1, 3, 320, 320)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0
    assert meta.original_width == 300


def test_to_tensor_swaps_bgr_to_rgb():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = 255
    tensor, _ = prepare_input(image, 32)
    assert tensor[0, 2].min() == 1.0
    assert tensor[0, 0].max() == 0.0
