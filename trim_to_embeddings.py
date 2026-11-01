"""Trim a BERT-MLM ONNX export down to a sentence-embedding model.

The bert-base-uncased export on the Triton host is BertForMaskedLM: the graph
ends in the 30522-way vocab decoder, so every request pays for a ~600 MB
[batch, seq, 30522] logits tensor. This script re-anchors the graph output on
the encoder's last_hidden_state and appends attention-masked mean pooling, so
the model returns [batch, 768] embeddings directly from the onnxruntime
backend (no Python wrapper, no 3 GB venv tarball).

Usage: python trim_to_embeddings.py <input.onnx> <output.onnx>
"""

import sys

import onnx
from onnx import TensorProto, helper

# Final encoder LayerNorm output == HF `last_hidden_state` for this export.
LAST_HIDDEN = "/bert/encoder/layer.11/output/LayerNorm/Add_1_output_0"
HIDDEN_SIZE = 768
POOL_OUTPUT = "EMBEDDING"


def build_pooling_nodes():
    """Mean-pool over the sequence axis, ignoring padded positions.

    The source graph is opset 11, where Unsqueeze/Squeeze/ReduceSum take `axes`
    as an *attribute* (it moved to an input in opset 13), so this sticks to the
    opset-11 signatures and needs no opset bump.
    """
    return [
        # attention_mask int64 [B, T] -> float [B, T, 1]
        helper.make_node(
            "Cast", ["attention_mask"], ["pool_mask_f32"],
            name="/pool/Cast", to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Unsqueeze", ["pool_mask_f32"], ["pool_mask"],
            name="/pool/Unsqueeze", axes=[2],
        ),
        # zero out padded tokens, then sum over the sequence axis
        helper.make_node(
            "Mul", [LAST_HIDDEN, "pool_mask"], ["pool_masked"],
            name="/pool/Mul",
        ),
        helper.make_node(
            "ReduceSum", ["pool_masked"], ["pool_sums"],
            name="/pool/ReduceSum", axes=[1], keepdims=1,
        ),
        helper.make_node(
            "ReduceSum", ["pool_mask"], ["pool_counts"],
            name="/pool/ReduceSum_counts", axes=[1], keepdims=1,
        ),
        # clamp the denominator so an all-padding row cannot divide by zero
        helper.make_node(
            "Max", ["pool_counts", "pool_eps"], ["pool_counts_safe"],
            name="/pool/Max",
        ),
        helper.make_node(
            "Div", ["pool_sums", "pool_counts_safe"], ["pool_pooled"],
            name="/pool/Div",
        ),
        # [B, 1, 768] -> [B, 768]
        helper.make_node(
            "Squeeze", ["pool_pooled"], [POOL_OUTPUT],
            name="/pool/Squeeze", axes=[1],
        ),
    ]


def build_constants():
    return [
        helper.make_tensor("pool_eps", TensorProto.FLOAT, [1], [1e-9]),
    ]


def main(src, dst):
    model = onnx.load(src)
    graph = model.graph

    if LAST_HIDDEN not in {n.output[0] for n in graph.node}:
        raise SystemExit(f"cannot find {LAST_HIDDEN}; graph layout changed")

    # Walk backwards from last_hidden_state and keep only the nodes it needs.
    # This drops the MLM head (cls/predictions/*), including the 30522x768
    # decoder weight and its transform/LayerNorm layers.
    producers = {}
    for node in graph.node:
        for out in node.output:
            producers[out] = node

    needed = {LAST_HIDDEN}
    stack = [LAST_HIDDEN]
    while stack:
        node = producers.get(stack.pop())
        if node is None:
            continue
        for name in node.input:
            if name not in needed:
                needed.add(name)
                stack.append(name)

    kept_nodes = [
        n for n in graph.node if n.output and n.output[0] in needed
    ]
    kept_inits = [i for i in graph.initializer if i.name in needed]

    # protobuf repeated fields have no slice assignment; clear then extend.
    del graph.node[:]
    graph.node.extend(kept_nodes + build_pooling_nodes())
    del graph.initializer[:]
    graph.initializer.extend(kept_inits + build_constants())
    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info(POOL_OUTPUT, TensorProto.FLOAT, [None, HIDDEN_SIZE])
    ])

    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"nodes {len(graph.node)}  initializers {len(graph.initializer)}  -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
