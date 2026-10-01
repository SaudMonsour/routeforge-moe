"""Trace exact byte-level expert choices through a trained decoder."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import argparse
import json
from pathlib import Path
import tensorflow as tf
from routeforge import load_model


def trace(model, raw):
    if not raw or len(raw) > model.config.context:
        raise ValueError("text must contain 1..context UTF-8 bytes")
    _, auxiliary, diagnostics = model.forward(tf.constant([list(raw)], tf.int32))
    if not model.config.experts:
        raise ValueError("routing trace requires a sparse model")
    rows = []
    for position, byte in enumerate(raw):
        rows.append({"position": position, "byte_id": byte, "hex": f"{byte:02x}",
                     "display": chr(byte) if 32 <= byte < 127 else f"\\x{byte:02x}",
                     "expert_ids": [int(d["assignments"][0, position]) for d in diagnostics],
                     "selected_probabilities": [float(d["selected_probability"][0, position]) for d in diagnostics]})
    return {"input_hex": raw.hex(), "auxiliary_loss": float(auxiliary), "tokens": rows,
            "counts": [d["counts"].numpy().astype(int).tolist() for d in diagnostics]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent/"runs/balanced")
    parser.add_argument("--text", default="The creature opened his eyes.")
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    result = trace(load_model(args.checkpoint), args.text.encode("utf-8"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
