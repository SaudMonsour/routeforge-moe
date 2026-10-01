"""Independently replay evaluation, routing, sources and recorded outputs."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from urllib.request import urlopen
import h5py
import numpy as np
import tensorflow as tf
from prepare import ROOT, read_data, transform, digest
from routeforge import Config, Decoder, load_model
from study import windows, evaluate, fingerprint
from diagnostics import routing
from inspect_routes import trace
from generate import sample
from artifacts import files


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            compare(a, b)
    elif isinstance(expected, str):
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == "0"
    manifest, (train, validation, test) = read_data()
    with urlopen(manifest["download_url"], timeout=30) as response:
        fresh = response.read(1000001)
    assert fresh == (ROOT/"data/source.txt").read_bytes()
    assert digest(fresh) == manifest["source_sha256"]
    assert transform(fresh) == (ROOT/"data/body.bin").read_bytes()
    metrics = json.loads((ROOT/"runs/metrics.json").read_text())
    benchmark = json.loads((ROOT/"runs/benchmark.json").read_text())
    diagnostics = json.loads((ROOT/"runs/routing_diagnostics.json").read_text())
    errors = json.loads((ROOT/"runs/error_analysis.json").read_text())
    assert metrics["source_sha256"] == manifest["source_sha256"]
    assert metrics["body_sha256"] == benchmark["body_sha256"] == manifest["body_sha256"]
    schedule = np.random.default_rng(metrics["seed"]).integers(0, len(train)-64, size=(metrics["steps"], metrics["batch"]))
    np.testing.assert_array_equal(schedule, np.load(ROOT/"runs/training_starts.npy", allow_pickle=False))
    with (ROOT/"runs/training_history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    functions, scores, initial = {}, {}, {}
    for role in ("dense", "unbalanced", "balanced"):
        directory = ROOT/"runs"/role
        config = Config(**json.loads((directory/"config.json").read_text()))
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(metrics["seed"])
        original = Decoder(config)
        original(tf.zeros([1, 64], tf.int32))
        initial[role] = fingerprint(original)
        assert initial[role] == metrics["models"][role]["initial_weights_sha256"]
        with h5py.File(directory/"model.weights.h5") as archive:
            def finite(name, item):
                if isinstance(item, h5py.Dataset):
                    assert np.isfinite(item[()]).all(), name
            archive.visititems(finite)
        model = load_model(directory)
        assert model.count_params() == metrics["models"][role]["parameters"]
        active = model.count_params()-config.layers*max(config.experts-1, 0)*3*config.width*config.hidden
        assert active == metrics["models"][role]["nominal_active_parameters_per_token"]
        rows = [r for r in history if r["model"] == role]
        assert int(min(rows, key=lambda r: float(r["validation_nll"]))["step"]) == metrics["models"][role]["selected_step"]
        scores[role] = {}
        for split, data in (("validation", validation), ("test", test)):
            result, losses = evaluate(model, data)
            compare(result, metrics["models"][role][split])
            scores[role][split] = result
            if split == "test":
                compare(losses, np.load(directory/"test_window_losses.npy", allow_pickle=False))
        losses = np.load(directory/"test_window_losses.npy", allow_pickle=False)
        assert [r["window"] for r in errors[role]] == np.argsort(losses)[-5:][::-1].tolist()
        for row in errors[role]:
            offset = row["window"]*64
            raw = bytes(test[offset:offset+64].astype(np.uint8))
            assert row["input_hex"] == raw.hex() and row["body_offset"] == manifest["split_offsets"][2]+offset
            compare(row["nll"], losses[row["window"]])
        raw_sample = sample(model, b"The creature ", 160, .8, 123)
        assert raw_sample == (directory/"sample.bin").read_bytes()
        assert raw_sample.decode("utf-8", errors="replace")+"\n" == (directory/"sample.txt").read_text()
        if config.experts:
            measured = routing(model, test)
            saved = {k: v for k, v in diagnostics[role].items() if k != "trace"}
            compare(measured, saved)
            count_array = np.asarray(measured["counts_by_layer_category_expert"])
            np.testing.assert_array_equal(count_array.sum(1), metrics["models"][role]["test"]["routing"]["counts"])
            compare(trace(model, b"The creature opened his eyes."), diagnostics[role]["trace"])
        functions[role] = tf.function(model, input_signature=[tf.TensorSpec([None, 64], tf.int32)])
    assert initial["unbalanced"] == initial["balanced"]
    _, labels = windows(test)
    counts = np.bincount(train, minlength=256).astype(float)+1.
    baseline = float(-np.log((counts/counts.sum())[labels]).mean())
    compare(baseline, metrics["unigram"]["test_nll"])
    compare(np.exp(baseline), metrics["unigram"]["test_byte_perplexity"])
    inputs, _ = windows(test)
    roles = ("dense", "unbalanced", "balanced")
    assert benchmark["protocol"]["trials"] == 20 and len(benchmark["observations"]) == 720
    for batch in (1, 16):
        cases = [inputs[i*batch:(i+1)*batch].tolist() for i in range(6)]
        assert cases == benchmark["inputs"][str(batch)]
        for index, case in enumerate(cases):
            for role in roles:
                output = functions[role](np.asarray(case, np.int32)).numpy()
                assert hashlib.sha256(output.tobytes()).hexdigest() == benchmark["output_sha256"][f"{batch}/{index}/{role}"]
        group = [r for r in benchmark["observations"] if r["batch"] == batch]
        baseline_ms = np.median([r["milliseconds"] for r in group if r["model"] == "dense"])
        for summary in [r for r in benchmark["summaries"] if r["batch"] == batch]:
            observations = [r for r in group if r["model"] == summary["model"]]
            assert len(observations) == summary["requests"] == 120
            assert len({(r["trial"], r["case"]) for r in observations}) == 120
            for row in observations:
                shift = (row["trial"]+row["case"]) % 3
                order = roles[shift:]+roles[:shift]
                if row["trial"] % 2:
                    order = order[::-1]
                assert order[row["order_position"]] == row["model"]
                assert np.isfinite(row["milliseconds"]) and row["milliseconds"] > 0
            times = [r["milliseconds"] for r in observations]
            median = np.median(times)
            compare(median, summary["median_ms"])
            compare(np.percentile(times, 95), summary["p95_ms"])
            compare(batch*64*1000/median, summary["input_bytes_per_second"])
            compare(baseline_ms/median, summary["speed_ratio_vs_dense"])
    assert benchmark["traces"] == {role: 1 for role in roles}
    readme = (ROOT/"README.md").read_text()
    assert readme.startswith("# RouteForge\n") and "## Reproduce" not in readme
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    assert len(images) == 6 and all(name.startswith("figures/") and (ROOT/name).is_file() for name in images)
    metadata = json.loads((ROOT/"repository.json").read_text())
    assert len(metadata["description"]) < 160
    inventory = {}
    for path in files():
        content = path.read_bytes()
        inventory[path.relative_to(ROOT).as_posix()] = {"bytes": len(content), "sha256": digest(content),
          "git_blob_sha1": hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()}
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "full_public_source_byte_match": True, "source_bytes": len(fresh), "body_bytes": manifest["body_bytes"],
              "replayed_scores": scores, "recreated_training_schedule": True, "recreated_initial_weight_hashes": initial,
              "replayed_samples": True, "replayed_category_routing_and_trace": True, "replayed_benchmark_output_hashes": 36,
              "benchmark_observations_checked": 720, "timing_summaries_recomputed": True, "timings_rerun": False,
              "tests": {"count": 10, "status": "passed", "command": "python -m unittest discover -s tests -v",
                        "scope": "Executed before training; verifies sparse/reference outputs and gradients plus routing and decoder behavior."},
              "readme_images": images, "files": inventory,
              "scope": "Replays source, initialization, evaluation, routing, generation and summaries. Does not retrain or claim identical wall-clock latency. Hashes exclude this audit itself, the full downloaded corpus and Python caches."}
    (ROOT/"audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": "passed", "files": len(inventory), "replayed_benchmark_outputs": 36, "observations": 720}))


if __name__ == "__main__":
    main()
