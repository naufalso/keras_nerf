import pytest
import tensorflow as tf
import numpy as np

from keras_nerf.data.rays import RaysGenerator


@pytest.fixture
def rays_generator():
    return RaysGenerator(
        focal_length=138.88887889922103,
        image_width=100,
        image_height=100,
        near=2.0,
        far=6.0,
        n_sample=32
    )


@pytest.fixture
def camera_params():
    return tf.constant([
        [
            -0.9999021887779236,
            0.004192245192825794,
            -0.013345719315111637,
            -0.05379832163453102
        ],
        [
            -0.013988681137561798,
            -0.2996590733528137,
            0.95394366979599,
            3.845470428466797
        ],
        [
            -4.656612873077393e-10,
            0.9540371894836426,
            0.29968830943107605,
            1.2080823183059692
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ], dtype=tf.float32)


def test_ray_generator_output_shape(rays_generator, camera_params):
    ray_origin, ray_direction, sample_points = rays_generator(camera_params)
    assert ray_origin.shape == (100, 100, 3)
    assert ray_direction.shape == (100, 100, 3)
    assert sample_points.shape == (100, 100, 32)

    assert ray_origin.dtype == tf.float32
    assert ray_direction.dtype == tf.float32
    assert sample_points.dtype == tf.float32

    rays = (ray_origin[..., None, :] +
            ray_direction[..., None, :] * sample_points[..., None])

    assert rays.shape == (100, 100, 32, 3)
