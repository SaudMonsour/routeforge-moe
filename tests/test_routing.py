import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import tensorflow as tf
from routeforge import Config, Decoder, SparseExperts, load_model


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.experimental.enable_op_determinism()

    def layer(self):
        tf.keras.utils.set_random_seed(42)
        layer = SparseExperts(4, 8, 4)
        layer(tf.eye(4))
        layer.router.kernel.assign(tf.eye(4)*2)
        return layer

    def test_dispatch_equals_explicit_all_expert_reference(self):
        layer = self.layer()
        x = tf.random.normal([3, 7, 4])
        actual, _, diagnostic = layer(x)
        flat = tf.reshape(x, [-1, 4])
        probability = tf.nn.softmax(layer.router(flat), -1)
        ids = tf.argmax(probability, -1, output_type=tf.int32)
        all_outputs = tf.stack([expert(flat) for expert in layer.experts], axis=1)
        selected = tf.gather(all_outputs, ids, axis=1, batch_dims=1)
        selected *= tf.gather(probability, ids, axis=1, batch_dims=1)[:, None]
        np.testing.assert_allclose(actual.numpy(), tf.reshape(selected, tf.shape(x)).numpy(), atol=1e-6)
        self.assertEqual(int(tf.reduce_sum(diagnostic["counts"])), 21)

    def test_sparse_and_reference_gradients_match(self):
        layer = self.layer()
        x = tf.eye(4)+.03
        def reference():
            probability = tf.nn.softmax(layer.router(x), -1)
            ids = tf.argmax(probability, -1, output_type=tf.int32)
            outputs = tf.stack([expert(x) for expert in layer.experts], 1)
            return tf.gather(outputs, ids, axis=1, batch_dims=1)*tf.gather(probability, ids, axis=1, batch_dims=1)[:, None]
        with tf.GradientTape() as tape:
            sparse_loss = tf.reduce_sum(layer(x)[0]**2)
        sparse = tape.gradient(sparse_loss, layer.trainable_variables)
        with tf.GradientTape() as tape:
            full_loss = tf.reduce_sum(reference()**2)
        full = tape.gradient(full_loss, layer.trainable_variables)
        for a, b in zip(sparse, full):
            self.assertIsNotNone(a)
            np.testing.assert_allclose(tf.convert_to_tensor(a).numpy(), tf.convert_to_tensor(b).numpy(), atol=1e-6)
        self.assertGreater(float(tf.norm(sparse[0])), 0)

    def test_empty_experts_are_safe_and_no_token_is_dropped(self):
        layer = self.layer()
        layer.router.kernel.assign(tf.zeros_like(layer.router.kernel))
        x = tf.ones([2, 3, 4])
        actual, _, diagnostic = layer(x)
        np.testing.assert_array_equal(diagnostic["counts"], [6, 0, 0, 0])
        np.testing.assert_allclose(actual, layer.experts[0](x)/4, atol=1e-6)
        self.assertTrue(np.isfinite(actual).all())

    def test_dispatch_restores_original_token_order(self):
        layer = self.layer()
        x = tf.eye(4)
        permutation = [3, 0, 2, 1]
        actual = layer(tf.gather(x, permutation))[0]
        np.testing.assert_allclose(actual, tf.gather(layer(x)[0], permutation), atol=1e-6)

    def test_auxiliary_loss_matches_equation_and_has_router_gradient(self):
        layer = self.layer()
        x = tf.constant([[1., 0., 0., 0.]]*5+[[0., 1., 0., 0.]])
        with tf.GradientTape() as tape:
            _, balance, diagnostic = layer(x)
        gradients = tape.gradient(balance, layer.router.trainable_variables)
        expected = 4*tf.reduce_sum(diagnostic["counts"]/6*diagnostic["mean_probability"])
        self.assertAlmostEqual(float(balance), float(expected), places=6)
        self.assertGreater(float(tf.norm(gradients[0])), 0)

    def decoder(self, experts=4):
        tf.keras.utils.set_random_seed(42)
        return Decoder(Config(vocabulary=16, context=8, width=16, heads=2, hidden=24, experts=experts))

    def test_causal_attention_blocks_future_tokens(self):
        model = self.decoder()
        a = model(tf.constant([[0, 1, 2, 3]]))
        b = model(tf.constant([[0, 1, 7, 8]]))
        np.testing.assert_allclose(a.numpy()[:, :2], b.numpy()[:, :2], atol=2e-6)

    def test_logits_are_independent_of_other_batch_rows(self):
        model = self.decoder()
        a = model(tf.constant([[0, 1, 2, 3]]))
        b = model(tf.constant([[0, 1, 2, 3], [4, 5, 6, 7]]))
        np.testing.assert_allclose(a.numpy()[0], b.numpy()[0], atol=2e-6)

    def test_checkpoint_roundtrip(self):
        model = self.decoder()
        ids = tf.constant([[0, 1, 2, 3]])
        expected = model(ids).numpy()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"config.json").write_text(json.dumps(asdict(model.config)))
            model.save_weights(root/"model.weights.h5")
            np.testing.assert_array_equal(expected, load_model(root)(ids).numpy())

    def test_dense_model_has_no_routing_loss(self):
        model = self.decoder(0)
        logits, balance, diagnostics = model.forward(tf.constant([[0, 1, 2]]))
        self.assertEqual(logits.shape, (1, 3, 16))
        self.assertEqual(float(balance), 0)
        self.assertTrue(all(d == {} for d in diagnostics))

    def test_invalid_dimensions_and_tokens_fail(self):
        for kwargs in ({"heads": 3}, {"experts": -1}, {"context": 0}, {"auxiliary_weight": -1}):
            with self.assertRaises(ValueError):
                Config(**kwargs)
        model = self.decoder()
        for ids in ([[16]], [[-1]], [[0]*9]):
            with self.assertRaises(tf.errors.InvalidArgumentError):
                model(tf.constant(ids))


if __name__ == "__main__":
    unittest.main()
