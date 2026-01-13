import numpy as np
import pytest

from sentinel.detection.onnx_engine import OnnxFaceDetector, static_input_size
from tests.test_bench import build_tiny_model


def test_static_input_size_prefers_the_model_shape():
    assert static_input_size([1, 3, 320, 320], 416) == 320


def test_dynamic_axes_fall_back_to_requested_size():
    assert static_input_size([1, 3, "height", "width"], 416) == 416


def test_non_square_model_is_rejected():
    with pytest.raises(ValueError):
        static_input_size([1, 3, 320, 416], 416)


def test_detector_adopts_input_size_of_the_loaded_model(tmp_path):
    build_tiny_model(tmp_path / "tiny.onnx", size=32)
    detector = OnnxFaceDetector(tmp_path / "tiny.onnx", input_size=416, intra_op_threads=1)
    assert detector.input_size == 32
    assert detector.input_name == "images"


def build_constant_detector_model(path, size=32):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    predictions = np.array([[[16.0, 2.0], [16.0, 2.0], [10.0, 2.0], [10.0, 2.0], [0.9, 0.1]]], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
            helper.make_node("Mul", ["mean", "zero"], ["silenced"]),
            helper.make_node("Add", ["silenced", "predictions"], ["output0"]),
        ],
        "constant-detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 2])],
        initializer=[
            numpy_helper.from_array(predictions, "predictions"),
            numpy_helper.from_array(np.zeros((), dtype=np.float32), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_detect_maps_boxes_back_to_the_original_frame(tmp_path):
    build_constant_detector_model(tmp_path / "const.onnx")
    detector = OnnxFaceDetector(tmp_path / "const.onnx", input_size=32, conf_threshold=0.5, intra_op_threads=1)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert len(detections) == 1
    assert detections[0].box == pytest.approx((22.0, 22.0, 42.0, 42.0))
    assert detections[0].score == pytest.approx(0.9)


def test_detect_applies_letterbox_for_wide_frames(tmp_path):
    build_constant_detector_model(tmp_path / "const.onnx")
    detector = OnnxFaceDetector(tmp_path / "const.onnx", input_size=32, conf_threshold=0.5, intra_op_threads=1)
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    box = detector.detect(frame)[0].box
    assert box == pytest.approx((22.0, 6.0, 42.0, 26.0))
