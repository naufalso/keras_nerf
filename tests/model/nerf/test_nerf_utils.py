import pytest
import tensorflow as tf

from keras_nerf.model.nerf.utils import NeRFUtils


@pytest.fixture
def nerf_utils():
    return NeRFUtils(
        batch_size=2,
        image_height=128,
        image_width=128,
        ray_chunks=1024,
        pos_emb_xyz=10,
        pos_emb_dir=4,
        white_background=True
    )


@pytest.fixture
def n_coarse():
    return 32


@pytest.fixture
def n_fine():
    return 64


@pytest.fixture
def pos_embedding_xyz():
    return 10


@pytest.fixture
def pos_embedding_dir():
    return 4


def test_render_image_depth(nerf_utils, n_coarse):
    # Generate random rgb, sigma, and points
    rgb = tf.random.uniform(shape=(2, 128, 128, n_coarse, 3))
    sigma = tf.random.uniform(shape=(2, 128, 128, n_coarse, 1))
    points = tf.random.uniform(shape=(2, 128, 128, n_coarse))

    image, depth, weights = nerf_utils.render_image_depth(
        rgb, sigma, points)

    assert image.shape == (2, 128, 128, 3)
    assert depth.shape == (2, 128, 128)
    assert weights.shape == (2, 128, 128, n_coarse)


def test_positional_encoding(nerf_utils, n_coarse, pos_embedding_xyz):
    # Generate random rays
    rays = tf.random.uniform(shape=(2, 128, 128, n_coarse, 3))

    # Positional encode the rays
    pos_encoded_rays = nerf_utils.positional_encoding(rays, pos_embedding_xyz)

    assert pos_encoded_rays.shape == (
        2, 128, 128, n_coarse, 3 * 2 * pos_embedding_xyz + 3)


def test_fine_hierarchical_sampling(nerf_utils, n_coarse, n_fine):
    # Generate random mid_points and weights
    coarse_points = tf.random.uniform(shape=(2, 128, 128, n_coarse))
    mid_points = 0.5 * (coarse_points[..., 1:] + coarse_points[..., :-1])

    weights = tf.random.uniform(shape=(2, 128, 128, n_coarse))

    # Apply hierarchical sampling
    fine_points = nerf_utils.fine_hierarchical_sampling(
        mid_points, weights, n_fine)

    assert fine_points.shape == (2, 128, 128, n_fine)


def test_encode_position_and_directions(nerf_utils, n_coarse, pos_embedding_xyz, pos_embedding_dir):
    ray_origin = tf.random.uniform(shape=(2, 128, 128, 3))
    ray_direction = tf.random.uniform(shape=(2, 128, 128, 3))
    sample_points = tf.random.uniform(shape=(2, 128, 128, n_coarse))

    # Encode the position and directions
    pos_encoded_rays, pos_encoded_directions = nerf_utils.encode_position_and_directions(
        ray_origin, ray_direction, sample_points)

    assert pos_encoded_rays.shape == (
        2, 128, 128, n_coarse, 3 * 2 * pos_embedding_xyz + 3)
    assert pos_encoded_directions.shape == (
        2, 128, 128, n_coarse, 3 * 2 * pos_embedding_dir + 3)


def test_encode_position_and_directions_chunk(nerf_utils, n_coarse, pos_embedding_xyz, pos_embedding_dir):
    ray_origin = tf.random.uniform(shape=(2, 128, 128, 3))
    ray_direction = tf.random.uniform(shape=(2, 128, 128, 3))
    sample_points = tf.random.uniform(shape=(2, 128, 128, n_coarse))

    ray_origin = tf.reshape(ray_origin, (-1, 3))
    ray_direction = tf.reshape(ray_direction, (-1, 3))
    sample_points = tf.reshape(sample_points, (-1, n_coarse))

    # Encode the position and directions
    pos_encoded_rays, pos_encoded_directions = nerf_utils.encode_position_and_directions(
        ray_origin, ray_direction, sample_points)

    assert pos_encoded_rays.shape == (
        2 * 128 * 128, n_coarse, 3 * 2 * pos_embedding_xyz + 3)
    assert pos_encoded_directions.shape == (
        2 * 128 * 128, n_coarse, 3 * 2 * pos_embedding_dir + 3)


def test_render_image_depth_chunk(nerf_utils, n_coarse):
    # Generate random rgb, sigma, and points
    rgb = tf.random.uniform(shape=(1024, n_coarse, 3))
    sigma = tf.random.uniform(shape=(1024, n_coarse, 1))
    points = tf.random.uniform(shape=(1024, n_coarse))

    image, depth, weights = nerf_utils.render_image_depth_chunk(
        rgb, sigma, points)

    assert image.shape == (1024, 3)
    assert depth.shape == (1024)
    assert weights.shape == (1024, n_coarse)
