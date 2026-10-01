"""Collect actual expert choices by input byte category and fixed samples."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import json
import numpy as np
import tensorflow as tf
from prepare import ROOT, read_data
from routeforge import load_model
from study import windows
from generate import sample
from inspect_routes import trace

CATEGORIES = ("ASCII letters", "Whitespace", "Digits", "ASCII punctuation", "Other bytes")


def category(byte):
    if 65 <= byte <= 90 or 97 <= byte <= 122:
        return 0
    if byte in (9, 10, 13, 32):
        return 1
    if 48 <= byte <= 57:
        return 2
    if 33 <= byte <= 126:
        return 3
    return 4


def routing(model, piece):
    inputs, _ = windows(piece)
    counts = np.zeros([model.config.layers, len(CATEGORIES), model.config.experts], dtype=np.int64)
    categories = np.array([[category(int(byte)) for byte in row] for row in inputs])
    @tf.function(input_signature=[tf.TensorSpec([None, 64], tf.int32)])
    def forward(ids):
        return model.forward(ids)[2]
    for offset in range(0, len(inputs), 16):
        diagnostics = forward(inputs[offset:offset+16])
        groups = categories[offset:offset+16].ravel()
        for layer, diagnostic in enumerate(diagnostics):
            ids = diagnostic["assignments"].numpy().ravel()
            np.add.at(counts[layer], (groups, ids), 1)
    return {"categories": list(CATEGORIES), "counts_by_layer_category_expert": counts.tolist(),
            "input_tokens_per_layer": int(inputs.size),
            "scope": "Descriptive input-byte categories, not semantic expert labels or causal evidence of specialization."}


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    _, (_, _, test) = read_data()
    result = {}
    for role in ("dense", "unbalanced", "balanced"):
        model = load_model(ROOT/"runs"/role)
        raw = sample(model, b"The creature ", 160, .8, 123)
        (ROOT/"runs"/role/"sample.bin").write_bytes(raw)
        (ROOT/"runs"/role/"sample.txt").write_text(raw.decode("utf-8", errors="replace")+"\n")
        if model.config.experts:
            result[role] = routing(model, test)
            result[role]["trace"] = trace(model, b"The creature opened his eyes.")
    (ROOT/"runs/routing_diagnostics.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"models": list(result), "routing_tokens_per_layer": result["balanced"]["input_tokens_per_layer"]}))


if __name__ == "__main__":
    main()
