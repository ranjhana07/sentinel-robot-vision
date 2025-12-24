from pathlib import Path

import onnx
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from sentinel.export.calibration import ImageFolderCalibrationReader

CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


def model_input_name(model_path: Path | str) -> str:
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {tensor.name for tensor in model.graph.initializer}
    names = [value.name for value in model.graph.input if value.name not in initializers]
    return names[0]


def quantize_dynamic_int8(
    model_path: Path | str,
    output_path: Path | str,
    per_channel: bool = False,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(model_path),
        str(output),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    return output


def head_nodes_to_exclude(model_path: Path | str) -> list[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    producers = {tensor: node for node in model.graph.node for tensor in node.output}
    pending = [producers[value.name] for value in model.graph.output if value.name in producers]
    excluded: list[str] = []
    seen: set[str] = set()
    while pending:
        node = pending.pop()
        if node.name in seen:
            continue
        seen.add(node.name)
        excluded.append(node.name)
        if node.op_type == "Conv" and "dfl" not in node.name:
            continue
        pending.extend(producers[tensor] for tensor in node.input if tensor in producers)
    return excluded


def preprocess_for_quantization(model_path: Path | str, output_path: Path | str) -> Path:
    from onnxruntime.quantization.shape_inference import quant_pre_process

    output = Path(output_path)
    quant_pre_process(str(model_path), str(output), skip_symbolic_shape=True)
    return output


def quantize_static_int8(
    model_path: Path | str,
    output_path: Path | str,
    calibration_dir: Path | str,
    input_size: int = 416,
    max_images: int = 200,
    method: str = "minmax",
    per_channel: bool = True,
    preprocess: bool = True,
    exclude_head: bool = True,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reader = ImageFolderCalibrationReader(
        calibration_dir,
        input_name=model_input_name(model_path),
        input_size=input_size,
        max_images=max_images,
    )
    source_path = Path(model_path)
    if preprocess:
        source_path = preprocess_for_quantization(model_path, output.with_suffix(".prep.onnx"))
    quantize_static(
        str(source_path),
        str(output),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CALIBRATION_METHODS[method],
        nodes_to_exclude=head_nodes_to_exclude(model_path) if exclude_head else [],
    )
    if preprocess:
        source_path.unlink(missing_ok=True)
    return output
