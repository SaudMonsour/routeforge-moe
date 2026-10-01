"""Top-one dispatch with no dropped tokens and differentiable gate weights."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import tensorflow as tf


@dataclass(frozen=True)
class Config:
    vocabulary: int = 256
    context: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2
    hidden: int = 128
    experts: int = 4
    auxiliary_weight: float = .01

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for k, v in asdict(self).items() if k not in {"experts", "auxiliary_weight"}):
            raise ValueError("positive integer dimensions required")
        if type(self.experts) is not int or self.experts < 0 or self.width % self.heads:
            raise ValueError("nonnegative expert count and divisible head width required")
        if not 0 <= self.auxiliary_weight <= 1:
            raise ValueError("auxiliary weight must be in [0, 1]")


class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, width, hidden):
        super().__init__()
        self.up = tf.keras.layers.Dense(hidden, use_bias=False)
        self.gate = tf.keras.layers.Dense(hidden, use_bias=False)
        self.down = tf.keras.layers.Dense(width, use_bias=False)
        self.width, self.hidden = width, hidden

    def build(self, shape):
        self.up.build([None, self.width])
        self.gate.build([None, self.width])
        self.down.build([None, self.hidden])
        super().build(shape)

    def call(self, inputs):
        return self.down(self.up(inputs)*tf.nn.silu(self.gate(inputs)))


class SparseExperts(tf.keras.layers.Layer):
    """Dispatch each token to exactly one expert; restore its original position.

    Unlike a dense stack of every expert's output, only selected token rows
    enter each expert. The selected softmax probability multiplies its output;
    normalizing a top-one gate to 1 would remove its task-loss gradient.
    """
    def __init__(self, width, hidden, experts=4):
        super().__init__()
        if min(width, hidden, experts) < 1:
            raise ValueError("positive dimensions required")
        self.width, self.n = width, experts
        self.router = tf.keras.layers.Dense(experts, use_bias=False)
        self.experts = [SwiGLU(width, hidden) for _ in range(experts)]

    def build(self, shape):
        self.router.build([None, self.width])
        for expert in self.experts:
            expert.build([None, self.width])
        super().build(shape)

    def call(self, inputs):
        shape = tf.shape(inputs)
        flat = tf.reshape(inputs, [-1, self.width])
        tf.debugging.assert_positive(tf.shape(flat)[0])
        probabilities = tf.nn.softmax(self.router(flat), axis=-1)
        assignments = tf.argmax(probabilities, -1, output_type=tf.int32)
        selected = tf.gather(probabilities, assignments, axis=1, batch_dims=1)
        positions = tf.dynamic_partition(tf.range(tf.shape(flat)[0]), assignments, self.n)
        pieces = []
        for expert, indices in zip(self.experts, positions):
            # Variables are built before tracing; an empty expert skips matmuls.
            rows = tf.gather(flat, indices)
            output = tf.cond(tf.size(indices) > 0, lambda: expert(rows),
                             lambda: tf.zeros([0, self.width], inputs.dtype))
            pieces.append(output*tf.gather(selected, indices)[:, None])
        combined = tf.dynamic_stitch(positions, pieces)
        counts = tf.math.bincount(assignments, minlength=self.n, maxlength=self.n, dtype=tf.float32)
        fractions = counts/tf.cast(tf.shape(flat)[0], tf.float32)
        mean_probability = tf.reduce_mean(probabilities, 0)
        balance = self.n*tf.reduce_sum(tf.stop_gradient(fractions)*mean_probability)
        diagnostics = {"counts": counts, "mean_probability": mean_probability,
                       "entropy": tf.reduce_mean(-tf.reduce_sum(probabilities*tf.math.log(tf.maximum(probabilities, 1e-9)), -1)),
                       "assignments": tf.reshape(assignments, shape[:-1]), "selected_probability": tf.reshape(selected, shape[:-1])}
        return tf.reshape(combined, shape), balance, diagnostics


class Attention(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = tf.keras.layers.Dense(3*config.width, use_bias=False)
        self.output_projection = tf.keras.layers.Dense(config.width, use_bias=False)

    def call(self, inputs):
        c = self.config
        batch, length = tf.shape(inputs)[0], tf.shape(inputs)[1]
        packed = tf.reshape(self.qkv(inputs), [batch, length, 3, c.heads, c.width//c.heads])
        q, k, v = tf.unstack(tf.transpose(packed, [2, 0, 3, 1, 4]), axis=0)
        scores = tf.matmul(q, k, transpose_b=True)*(c.width//c.heads)**-.5
        mask = tf.range(length)[None, :] <= tf.range(length)[:, None]
        scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
        output = tf.matmul(tf.nn.softmax(scores, -1), v)
        output = tf.reshape(tf.transpose(output, [0, 2, 1, 3]), [batch, length, c.width])
        return self.output_projection(output)


class Block(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attention = Attention(config)
        self.ffn = SparseExperts(config.width, config.hidden, config.experts) if config.experts else SwiGLU(config.width, config.hidden)

    def call(self, inputs):
        hidden = inputs+self.attention(self.norm1(inputs))
        if self.config.experts:
            delta, loss, diagnostic = self.ffn(self.norm2(hidden))
        else:
            delta, loss, diagnostic = self.ffn(self.norm2(hidden)), tf.constant(0.), {}
        return hidden+delta, loss, diagnostic


class Decoder(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = tf.keras.layers.Embedding(config.vocabulary, config.width)
        self.position = tf.keras.layers.Embedding(config.context, config.width)
        self.blocks = [Block(config) for _ in range(config.layers)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def forward(self, ids):
        tf.debugging.assert_rank(ids, 2)
        tf.debugging.assert_positive(tf.shape(ids))
        tf.debugging.assert_less_equal(tf.shape(ids)[1], self.config.context)
        tf.debugging.assert_greater_equal(ids, 0)
        tf.debugging.assert_less(ids, self.config.vocabulary)
        hidden = self.embedding(ids)+self.position(tf.range(tf.shape(ids)[1]))[None]
        auxiliary, diagnostics = [], []
        for block in self.blocks:
            hidden, loss, diagnostic = block(hidden)
            auxiliary.append(loss)
            diagnostics.append(diagnostic)
        logits = tf.einsum("btd,vd->btv", self.norm(hidden), self.embedding.embeddings)
        # Mean per-layer loss: coefficient is independent of layer count.
        return logits, tf.reduce_mean(auxiliary), tuple(diagnostics)

    def call(self, ids, training=False):
        return self.forward(ids)[0]


def load_model(directory):
    root = Path(directory)
    model = Decoder(Config(**json.loads((root/"config.json").read_text())))
    model(tf.zeros([1, 1], tf.int32))
    model.load_weights(root/"model.weights.h5")
    return model
