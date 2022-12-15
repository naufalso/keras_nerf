import pytest
import tensorflow as tf
from keras_nerf.model.nerf.mlp import NeRFMLP


def test_nerf_mlp():
    nerf_mlp = NeRFMLP(
        n_layers=8,
        dense_units=256,
        skip_layer=4,
    )

    POS_ENCODE_DIMS = 16
    SAMPLE_POINTS = 32
    FINAL_POS_ENCODE_DIMS = 2 * 3 * POS_ENCODE_DIMS + 3

    ray_coordinate_inputs = tf.random.uniform(
        shape=(2, 100, 100, SAMPLE_POINTS, FINAL_POS_ENCODE_DIMS))
    direction_inputs = tf.random.uniform(
        shape=(2, 100, 100, SAMPLE_POINTS, FINAL_POS_ENCODE_DIMS))

    ray_coordinate_inputs = tf.reshape(
        ray_coordinate_inputs, (-1, SAMPLE_POINTS, FINAL_POS_ENCODE_DIMS))
    direction_inputs = tf.reshape(
        direction_inputs, (-1, SAMPLE_POINTS, FINAL_POS_ENCODE_DIMS))

    rgb_out, sigma_out = nerf_mlp((ray_coordinate_inputs, direction_inputs))

    model_params = nerf_mlp.get_config()

    assert model_params['n_layers'] == 8
    assert model_params['dense_units'] == 256
    assert model_params['skip_layer'] == 4

    assert rgb_out.shape == (2 * 100 * 100, SAMPLE_POINTS, 3)
    assert sigma_out.shape == (2 * 100 * 100, SAMPLE_POINTS, 1)

    rgb_out = tf.reshape(rgb_out, (2, 100, 100, SAMPLE_POINTS, 3))
    sigma_out = tf.reshape(sigma_out, (2, 100, 100, SAMPLE_POINTS, 1))

    assert rgb_out.shape == (2, 100, 100, SAMPLE_POINTS, 3)
    assert sigma_out.shape == (2, 100, 100, SAMPLE_POINTS, 1)

    assert tf.concat(nerf_mlp((ray_coordinate_inputs, direction_inputs)), axis=-1).shape == (
        2 * 100 * 100, SAMPLE_POINTS, 4)
