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


def quantize_static_int8(
    model_path: Path | str,
    output_path: Path | str,
    calibration_dir: Path | str,
    input_size: int = 416,
    max_images: int = 200,
    method: str = "minmax",
    per_channel: bool = True,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reader = ImageFolderCalibrationReader(
        calibration_dir,
        input_name=model_input_name(model_path),
        input_size=input_size,
        max_images=max_images,
    )
    quantize_static(
        str(model_path),
        str(output),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CALIBRATION_METHODS[method],
    )
    return output
