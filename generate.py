"""Generate raw byte IDs with an included checkpoint; no external API."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
from routeforge import load_model


def sample(model, prompt, count=128, temperature=.8, seed=123):
    if not prompt or count < 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("nonempty prompt, nonnegative count and positive finite temperature required")
    context = model.config.context
    @tf.function(input_signature=[tf.TensorSpec([1, None], tf.int32)])
    def forward(ids):
        return model(ids)[:, -1]
    ids = list(prompt)
    rng = np.random.default_rng(seed)
    for _ in range(count):
        logits = forward(np.array([ids[-context:]], np.int32)).numpy()[0].astype(np.float64)
        probability = np.exp((logits-logits.max())/temperature)
        ids.append(int(rng.choice(len(probability), p=probability/probability.sum())))
    return bytes(ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent/"runs/balanced")
    parser.add_argument("--prompt", default="The creature ")
    parser.add_argument("--bytes", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    print(sample(load_model(args.checkpoint), args.prompt.encode("utf-8"), args.bytes, args.temperature, args.seed).decode("utf-8", errors="replace"))
