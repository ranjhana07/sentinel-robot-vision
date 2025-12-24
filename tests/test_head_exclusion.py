import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from sentinel.export.quantize import head_nodes_to_exclude


def build(path):
    rng = np.random.default_rng(0)

    def conv_weights(name):
        return numpy_helper.from_array(rng.normal(size=(4, 4, 1, 1)).astype(np.float32), name)

    nodes = [
        helper.make_node("Conv", ["images", "w_backbone"], ["feat"], name="backbone_conv"),
        helper.make_node("Conv", ["feat", "w_pred_a"], ["raw_boxes"], name="pred_box_conv"),
        helper.make_node("Conv", ["feat", "w_pred_b"], ["raw_scores"], name="pred_cls_conv"),
        helper.make_node("Sigmoid", ["raw_scores"], ["scores"], name="scores_sigmoid"),
        helper.make_node("Concat", ["raw_boxes", "scores"], ["output0"], axis=1, name="final_concat"),
    ]
    graph = helper.make_graph(
        nodes,
        "head",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 4, 8, 8])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 8, 8, 8])],
        initializer=[conv_weights("w_backbone"), conv_weights("w_pred_a"), conv_weights("w_pred_b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_tail_and_prediction_convs_are_excluded_backbone_is_not(tmp_path):
    build(tmp_path / "head.onnx")
    excluded = set(head_nodes_to_exclude(tmp_path / "head.onnx"))
    assert excluded == {"final_concat", "scores_sigmoid", "pred_box_conv", "pred_cls_conv"}
    assert "backbone_conv" not in excluded
