import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from sentinel.export.quantize import model_input_name, quantize_dynamic_int8, quantize_static_int8


def save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def conv_model(path, size=32):
    rng = np.random.default_rng(1)
    weights = numpy_helper.from_array(rng.normal(size=(4, 3, 3, 3)).astype(np.float32), "w")
    bias = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "b")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["images", "w", "b"], ["features"], pads=[1, 1, 1, 1])],
        "conv",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 4, size, size])],
        initializer=[weights, bias],
    )
    save(graph, path)


def matmul_model(path):
    rng = np.random.default_rng(2)
    weights = numpy_helper.from_array(rng.normal(size=(16, 8)).astype(np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "matmul",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])],
        initializer=[weights],
    )
    save(graph, path)


def test_model_input_name_ignores_initializers(tmp_path):
    conv_model(tmp_path / "conv.onnx")
    assert model_input_name(tmp_path / "conv.onnx") == "images"


def test_static_quantization_produces_runnable_model(tmp_path):
    conv_model(tmp_path / "conv.onnx")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    rng = np.random.default_rng(3)
    for index in range(4):
        cv2.imwrite(str(calibration / f"{index}.png"), rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    output = quantize_static_int8(
        tmp_path / "conv.onnx", tmp_path / "conv-int8.onnx", calibration, input_size=32, max_images=4
    )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    sample = rng.random((1, 3, 32, 32), dtype=np.float32)
    quantized = session.run(None, {"images": sample})[0]
    reference = ort.InferenceSession(str(tmp_path / "conv.onnx"), providers=["CPUExecutionProvider"])
    expected = reference.run(None, {"images": sample})[0]
    assert quantized.shape == expected.shape
    assert np.abs(quantized - expected).mean() < 0.5
    assert output.stat().st_size > 0


def test_dynamic_quantization_produces_runnable_model(tmp_path):
    matmul_model(tmp_path / "mm.onnx")
    output = quantize_dynamic_int8(tmp_path / "mm.onnx", tmp_path / "out" / "mm-int8.onnx")
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(4).random((1, 16), dtype=np.float32)
    result = session.run(None, {"x": x})[0]
    assert result.shape == (1, 8)
    op_types = {node.op_type for node in onnx.load(str(output)).graph.node}
    assert "MatMulInteger" in op_types or "DynamicQuantizeLinear" in op_types


def test_static_quantization_needs_images(tmp_path):
    conv_model(tmp_path / "conv.onnx")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        quantize_static_int8(tmp_path / "conv.onnx", tmp_path / "o.onnx", tmp_path / "empty")


def test_static_quantization_without_preprocess_leaves_no_temp_files(tmp_path):
    conv_model(tmp_path / "conv.onnx")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    cv2.imwrite(str(calibration / "a.png"), np.full((40, 40, 3), 90, dtype=np.uint8))
    quantize_static_int8(
        tmp_path / "conv.onnx", tmp_path / "out.onnx", calibration, input_size=32, preprocess=False
    )
    assert sorted(path.name for path in tmp_path.glob("*.onnx")) == ["conv.onnx", "out.onnx"]


def test_static_quantization_cleans_up_preprocessed_copy(tmp_path):
    conv_model(tmp_path / "conv.onnx")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    cv2.imwrite(str(calibration / "a.png"), np.full((40, 40, 3), 90, dtype=np.uint8))
    quantize_static_int8(tmp_path / "conv.onnx", tmp_path / "out.onnx", calibration, input_size=32)
    assert sorted(path.name for path in tmp_path.glob("*.onnx")) == ["conv.onnx", "out.onnx"]
