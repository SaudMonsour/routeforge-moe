"""Counterbalanced full-sequence inference latency with actual sparse dispatch."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import hashlib
import json
import time
from datetime import datetime, timezone
import numpy as np
import tensorflow as tf
from prepare import ROOT, read_data
from study import windows
from routeforge import load_model

ROLES = ("dense", "unbalanced", "balanced")


def main():
    path = ROOT/"runs/benchmark.json"
    if path.exists():
        raise ValueError("preserve original benchmark observations")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    manifest, (_, _, test) = read_data()
    inputs, _ = windows(test)
    functions = {}
    for role in ROLES:
        model = load_model(ROOT/"runs"/role)
        functions[role] = tf.function(model, input_signature=[tf.TensorSpec([None, 64], tf.int32)])
    observations, checksums, groups = [], {}, {}
    started = datetime.now(timezone.utc).isoformat()
    for batch in (1, 16):
        cases = [inputs[i*batch:(i+1)*batch] for i in range(6)]
        groups[str(batch)] = [case.tolist() for case in cases]
        for role in ROLES:
            for index, case in enumerate(cases):
                for _ in range(3):
                    output = functions[role](case).numpy()
                checksums[f"{batch}/{index}/{role}"] = hashlib.sha256(output.tobytes()).hexdigest()
        for trial in range(20):
            for index, case in enumerate(cases):
                shift = (trial+index) % 3
                order = ROLES[shift:]+ROLES[:shift]
                if trial % 2:
                    order = order[::-1]
                for position, role in enumerate(order):
                    began = time.perf_counter_ns()
                    output = functions[role](case).numpy()
                    milliseconds = (time.perf_counter_ns()-began)/1e6
                    assert hashlib.sha256(output.tobytes()).hexdigest() == checksums[f"{batch}/{index}/{role}"]
                    observations.append({"batch": batch, "case": index, "trial": trial,
                                         "order_position": position, "model": role, "milliseconds": milliseconds})
    summaries = []
    for batch in (1, 16):
        baseline = np.median([r["milliseconds"] for r in observations if r["batch"] == batch and r["model"] == "dense"])
        for role in ROLES:
            values = [r["milliseconds"] for r in observations if r["batch"] == batch and r["model"] == role]
            median = float(np.median(values))
            summaries.append({"batch": batch, "model": role, "requests": len(values), "median_ms": median,
                              "p95_ms": float(np.percentile(values, 95)), "input_bytes_per_second": batch*64*1000/median,
                              "speed_ratio_vs_dense": float(baseline/median)})
    result = {"started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "body_sha256": manifest["body_sha256"], "protocol": {"batches": [1, 16], "context": 64, "cases": 6,
              "trials": 20, "warmups_per_case_model": 3, "order": "Rotate by trial+case; reverse on odd trials.",
              "scope": "Whole 64-byte sequence forward pass, including routing and CPU logit materialization; excludes loading and warmup. No KV cache or autoregressive request benchmark.",
              "runtime": "CPU float32; intra/inter threads 2/2; oneDNN disabled; shared host."},
              "inputs": groups, "output_sha256": checksums, "observations": observations,
              "summaries": summaries, "traces": {role: f.experimental_get_tracing_count() for role, f in functions.items()}}
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"summaries": summaries, "observations": len(observations), "traces": result["traces"]}))


if __name__ == "__main__":
    main()
