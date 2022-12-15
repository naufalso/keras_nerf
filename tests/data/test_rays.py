import pytest
import tensorflow as tf
import numpy as np

from keras_nerf.data.rays import RaysGenerator


@pytest.fixture
def rays_generator():
    return RaysGenerator(
        focal_length=138.88887889922103,
        image_width=128,
        image_height=128,
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
    last_ray_origin = None
    last_ray_direction = None
    last_sample_points = None

    for i in range(4):
        ray_origin, ray_direction, sample_points = rays_generator(
            camera_params)

        assert np.isnan(ray_origin).sum() == 0
        assert np.isnan(ray_direction).sum() == 0
        assert np.isnan(sample_points).sum() == 0

        assert ray_origin.shape == (128, 128, 3)
        assert ray_direction.shape == (128, 128, 3)
        assert sample_points.shape == (128, 128, 32)

        assert ray_origin.dtype == tf.float32
        assert ray_direction.dtype == tf.float32
        assert sample_points.dtype == tf.float32

        if last_sample_points is not None:
            assert np.allclose(last_ray_origin, ray_origin)
            assert np.allclose(last_ray_direction, ray_direction)
            assert np.allclose(last_sample_points,
                               sample_points, atol=(4.0 / 32.0))

        assert np.min(sample_points) >= 2.0 - (4.0 / 32.0) and np.max(
            sample_points) <= 6.0 + (4.0 / 32.0)

        rays = (ray_origin[..., None, :] +
                ray_direction[..., None, :] * sample_points[..., None])

        assert rays.shape == (128, 128, 32, 3)

        last_ray_origin = ray_origin.numpy().copy()
        last_ray_direction = ray_direction.numpy().copy()
        last_sample_points = sample_points.numpy().copy()
