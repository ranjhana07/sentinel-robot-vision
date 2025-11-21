import csv
import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from sentinel.bench import BenchResult, benchmark_model, latency_stats, render_markdown, write_reports


def build_tiny_model(path, size=32):
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["pooled"], keepdims=0)],
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("pooled", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_latency_stats():
    stats = latency_stats([10.0, 20.0, 30.0, 40.0])
    assert stats["mean_ms"] == pytest.approx(25.0)
    assert stats["p50_ms"] == pytest.approx(25.0)
    assert stats["fps"] == pytest.approx(40.0)
    assert stats["p95_ms"] > stats["p50_ms"]


def test_latency_stats_requires_samples():
    with pytest.raises(ValueError):
        latency_stats([])


def make_result(name, fps):
    return BenchResult(name, 10.0, 50, 1000.0 / fps, 1000.0 / fps, 1100.0 / fps, fps)


def test_markdown_reports_speedup_against_first_row():
    table = render_markdown([make_result("fp32.onnx", 5.0), make_result("int8.onnx", 10.0)])
    rows = table.strip().splitlines()
    assert rows[0].startswith("| model")
    assert rows[2].endswith("| 1.00x |")
    assert rows[3].endswith("| 2.00x |")


def test_write_reports_creates_all_files(tmp_path):
    markdown_path, csv_path, json_path = write_reports(
        [make_result("a.onnx", 8.0)], tmp_path / "out", {"threads": 4}, stem="run"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["settings"] == {"threads": 4}
    assert payload["environment"]["logical_cores"] >= 1
    assert payload["results"][0]["name"] == "a.onnx"
    assert markdown_path.name == "run.md"
    assert markdown_path.read_text(encoding="utf-8").count("a.onnx") == 1
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["name"] == "a.onnx"
    assert float(rows[0]["fps"]) == 8.0


def test_benchmark_model_on_tiny_graph(tmp_path):
    model_path = tmp_path / "tiny.onnx"
    build_tiny_model(model_path)
    result = benchmark_model(model_path, input_size=32, runs=5, warmup=1, threads=1)
    assert result.runs == 5
    assert result.mean_ms > 0
    assert np.isfinite(result.fps)
