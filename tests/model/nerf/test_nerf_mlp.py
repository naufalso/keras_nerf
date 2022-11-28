import pytest
import tensorflow as tf
from keras_nerf.model.nerf.mlp import NeRFMLP


def test_nerf_mlp():
    nerf_mlp = NeRFMLP()

    POS_ENCODE_DIMS = 16
    SAMPLE_POINTS = 32
    FINAL_POS_ENCODE_DIMS = 2 * 3 * POS_ENCODE_DIMS + 3

    ray_coordinate_inputs = tf.random.uniform(
        shape=(2, 100, 100, SAMPLE_POINTS, FINAL_POS_ENCODE_DIMS))
    direction_inputs = tf.random.uniform(
        shape=(2, 100, 100, SAMPLE_POINTS, FINAL_POS_ENCODE_DIMS))

    rgb_out, sigma_out = nerf_mlp((ray_coordinate_inputs, direction_inputs))

    nerf_mlp.summary()

    assert rgb_out.shape == (2, 100, 100, SAMPLE_POINTS, 3)
    assert sigma_out.shape == (2, 100, 100, SAMPLE_POINTS, 1)
