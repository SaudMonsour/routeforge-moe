"""Fixed-budget dense / sparse / balanced-sparse byte language study."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from routeforge import Config, Decoder, load_model
from prepare import ROOT, read_data


def windows(piece, context=64):
    starts = np.arange(0, len(piece)-context, context)
    indices = starts[:, None]+np.arange(context)[None]
    return piece[indices], piece[indices+1]


def evaluate(model, piece):
    inputs, labels = windows(piece, model.config.context)
    @tf.function(input_signature=[tf.TensorSpec([None, model.config.context], tf.int32)])
    def forward(ids):
        return model.forward(ids)
    losses, accuracy, counts, entropy, probabilities = [], [], None, [], None
    for offset in range(0, len(inputs), 16):
        logits, _, diagnostics = forward(inputs[offset:offset+16])
        target = labels[offset:offset+16]
        per_byte = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits).numpy()
        losses.extend(per_byte.mean(-1).tolist())
        accuracy.extend(np.mean(logits.numpy().argmax(-1) == target, -1).tolist())
        if model.config.experts:
            number = int(target.size)
            current = np.array([d["counts"].numpy() for d in diagnostics], dtype=np.int64)
            weighted = np.array([d["mean_probability"].numpy() for d in diagnostics])*number
            ent = np.array([float(d["entropy"])*number for d in diagnostics])
            counts = current if counts is None else counts+current
            probabilities = weighted if probabilities is None else probabilities+weighted
            entropy.append(ent)
    nll = float(np.mean(losses))
    result = {"nll": nll, "byte_perplexity": float(np.exp(nll)), "next_byte_accuracy": float(np.mean(accuracy)),
              "windows": len(inputs), "evaluated_bytes": int(inputs.size), "window_nll_sd": float(np.std(losses, ddof=1))}
    if counts is not None:
        total = int(inputs.size)
        assert np.all(counts.sum(-1) == total)
        result["routing"] = {"counts": counts.tolist(), "fractions": (counts/total).tolist(),
                             "mean_probability": (probabilities/total).tolist(), "mean_router_entropy_nats": (np.sum(entropy, 0)/total).tolist(),
                             "max_over_mean_load": (counts.max(-1)/(total/model.config.experts)).tolist(),
                             "unused_experts": (counts == 0).sum(-1).tolist(), "dropped_tokens": 0}
    return result, np.array(losses)


def fingerprint(model):
    h = hashlib.sha256()
    for variable in model.trainable_variables:
        value = variable.numpy()
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch) <= 0:
        parser.error("positive steps/batch required")
    output = ROOT/"runs"
    if output.exists():
        parser.error("preserve existing measurements: runs already exists")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    manifest, (train, validation, test) = read_data()
    output.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    schedule = np.random.default_rng(args.seed).integers(0, len(train)-64, size=(args.steps, args.batch))
    np.save(output/"training_starts.npy", schedule)
    configurations = {"dense": Config(experts=0, auxiliary_weight=0.),
                      "unbalanced": Config(auxiliary_weight=0.), "balanced": Config(auxiliary_weight=.01)}
    models, history = {}, []
    initial_sparse = None
    for name, config in configurations.items():
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        model = Decoder(config)
        model(tf.zeros([1, 64], tf.int32))
        initial = fingerprint(model)
        if name == "unbalanced":
            initial_sparse = initial
        if name == "balanced":
            assert initial == initial_sparse, "sparse ablation initialization differs"
        directory = output/name
        directory.mkdir()
        (directory/"config.json").write_text(json.dumps(asdict(config), indent=2)+"\n")
        optimizer = tf.keras.optimizers.Adam(.001, global_clipnorm=1.)
        optimizer.build(model.trainable_variables)
        @tf.function(input_signature=[tf.TensorSpec([args.batch, 64], tf.int32), tf.TensorSpec([args.batch, 64], tf.int32)])
        def train_step(ids, labels):
            with tf.GradientTape() as tape:
                logits, auxiliary, _ = model.forward(ids)
                nll = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                loss = nll+config.auxiliary_weight*auxiliary
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return nll, auxiliary
        best, best_step = float("inf"), 0
        began = time.perf_counter()
        for step, starts in enumerate(schedule, 1):
            indices = starts[:, None]+np.arange(64)[None]
            nll, auxiliary = train_step(train[indices], train[indices+1])
            if step == 1 or step % 100 == 0 or step == args.steps:
                val, _ = evaluate(model, validation)
                row = {"model": name, "step": step, "train_nll": float(nll), "train_auxiliary": float(auxiliary),
                       "validation_nll": val["nll"], "elapsed_seconds": time.perf_counter()-began}
                history.append(row)
                print(json.dumps(row), flush=True)
                if val["nll"] < best:
                    best, best_step = val["nll"], step
                    model.save_weights(directory/"model.weights.h5")
        model.load_weights(directory/"model.weights.h5")
        val, _ = evaluate(model, validation)
        total = model.count_params()
        expert_weights = config.layers*config.experts*3*config.width*config.hidden
        active = total-expert_weights+config.layers*3*config.width*config.hidden if config.experts else total
        models[name] = {"parameters": total, "nominal_active_parameters_per_token": active,
                        "active_parameter_note": "All shared weights plus one expert per layer; embeddings counted in full. This is not measured FLOPs or resident memory.",
                        "selected_step": best_step, "initial_weights_sha256": initial, "validation": val,
                        "training_seconds_with_validation_and_checkpoint_saving": time.perf_counter()-began}
    errors = {}
    for name in models:
        model = load_model(output/name)
        heldout, losses = evaluate(model, test)
        models[name]["test"] = heldout
        np.save(output/name/"test_window_losses.npy", losses)
        errors[name] = []
        for window in np.argsort(losses)[-5:][::-1]:
            offset = int(window)*64
            raw = bytes(test[offset:offset+64].astype(np.uint8))
            errors[name].append({"window": int(window), "body_offset": manifest["split_offsets"][2]+offset,
                                 "nll": float(losses[window]), "input_hex": raw.hex(), "preview": raw.decode("utf-8", errors="replace")})
    _, labels = windows(test)
    counts = np.bincount(train, minlength=256).astype(float)+1.
    baseline = float(-np.log((counts/counts.sum())[labels]).mean())
    result = {"started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "seed": args.seed, "steps": args.steps, "batch": args.batch, "sampled_byte_targets_per_model": args.steps*args.batch*64,
              "source_sha256": manifest["source_sha256"], "body_sha256": manifest["body_sha256"], "models": models,
              "unigram": {"test_nll": baseline, "test_byte_perplexity": float(np.exp(baseline))},
              "controls": "Identical seed, sampled windows/order, optimizer, learning rate, batch and steps. Sparse variants start with identical weights. Dense has fewer total parameters; one expert has the same width/hidden size as its dense FFN. No seed search, pretrained weights, distillation or test-driven selection.",
              "selection": "Independent minimum validation NLL checkpoint for each fixed architecture; test evaluated only after all selections.",
              "runtime": {"python": platform.python_version(), "tensorflow": tf.__version__, "platform": platform.platform(),
                          "intra_threads": 2, "inter_threads": 2, "oneDNN": False, "devices": [str(d) for d in tf.config.list_physical_devices()]}}
    (output/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    (output/"error_analysis.json").write_text(json.dumps(errors, indent=2)+"\n")
    with (output/"training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
