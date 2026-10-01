# Using RouteForge

The three included checkpoints and routing tools run locally without API keys, paid services or a corpus download. The executed environment used Python 3.12.14 and TensorFlow 2.20.0; exact dependency versions are in `environment.lock.txt`.

## Inspect expert choices

```bash
pip install -r requirements.txt
python inspect_routes.py --text "The creature opened his eyes."
python inspect_routes.py --checkpoint runs/unbalanced --text "The creature opened his eyes."
```

The JSON output contains byte IDs, hex values, each layer's expert ID, selected gate probability and expert counts. Expert IDs are local to a layer and model. An ID does not mean the model learned a named subject or a human-assigned skill.

Inputs must contain 1–64 UTF-8 bytes. For non-ASCII text, multiple byte IDs can represent one character; the trace deliberately displays non-printable bytes as hex escapes. It never treats a continuation byte as a complete character.

## Generate bytes

```bash
python generate.py --checkpoint runs/balanced --prompt "The creature " --bytes 128
python generate.py --checkpoint runs/dense --prompt "The creature " --bytes 128
```

The CLI samples at temperature `0.8` and seed `123` by default. It re-evaluates the latest 64-byte window for every next token; it does not provide a KV cache. Output can contain invalid UTF-8. The CLI uses replacement characters for display, while the Python `sample` function returns exact raw bytes. The models are small research decoders and do not follow instructions.

## Reuse the sparse layer

Install the library from the repository root:

```bash
pip install -e .
```

```python
import tensorflow as tf
from routeforge import SparseExperts

experts = SparseExperts(width=64, hidden=128, experts=4)
hidden = tf.random.normal([8, 32, 64])
output, balance_loss, diagnostics = experts(hidden)

print(output.shape)                 # (8, 32, 64)
print(diagnostics["counts"])        # assignments across this call's token rows
print(diagnostics["assignments"])   # shape (8, 32)
```

The layer returns the auxiliary term rather than silently registering a Keras model loss. Add it explicitly to your objective inside a gradient tape:

```python
with tf.GradientTape() as tape:
    output, balance_loss, diagnostics = experts(hidden)
    task_loss = tf.reduce_mean(tf.square(output - desired_output))
    loss = task_loss + 0.01 * balance_loss
gradients = tape.gradient(loss, experts.trainable_variables)
```

`desired_output` is your task's target tensor and must match `output` in this example. Use your own optimizer and task loss. The provided decoder averages the auxiliary term across layers before applying its coefficient.

The gate keeps the selected softmax probability. Normalizing a top-one gate to exactly one removes its differentiable task-loss path. The integer argmax itself has no gradient; the auxiliary term uses stopped-gradient assignment frequencies and differentiable mean router probabilities.

There is no fixed expert capacity, overflow policy or token dropping. All selected rows are processed. Every expert's weights remain resident, and dispatch is on one device. The layer is not an expert-parallel serving runtime. It accepts dense, nonempty floating-point hidden states with the configured final width; padding masks and mixed precision need separate implementations and validation.

## Study files

`prepare.py` verifies the complete public source against the manifest. `study.py` trains the three fixed architectures with the same sampled windows and refuses to overwrite existing `runs/`. `benchmark.py` preserves original timings; `diagnostics.py` collects routing categories and fixed samples. `plots.py` renders figures from measurements, and `verify.py` replays source bytes, scores, routing counts, samples and benchmark output hashes.

The original Gutenberg source may later change; an updated header alone can change its raw hash. Preserve `data/manifest.json` as the record of the executed source. Downloaded raw data is excluded from publication. Model generation and routing inspection do not depend on that source remaining unchanged.

Run the focused correctness suite with:

```bash
python -m unittest discover -s tests -v
```

For a new experiment, use a separate checkout or change the output location deliberately. Keep the published checkpoints and observations intact. Useful extensions include top-two routing, capacity controls with explicit overflow accounting, expert-parallel dispatch and routing tests with padding masks. Each changes the layer's semantics and requires a new comparison.
