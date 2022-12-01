import pytest
import tensorflow as tf

from keras_nerf.model.nerf.utils import render_image_depth, positional_encoding, fine_hierarchical_sampling, encode_position_and_directions


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
    return 10


def test_render_image_depth(n_coarse):
    # Generate random rgb, sigma, and points
    rgb = tf.random.uniform(shape=(2, 100, 100, n_coarse, 3))
    sigma = tf.random.uniform(shape=(2, 100, 100, n_coarse, 1))
    points = tf.random.uniform(shape=(2, 100, 100, n_coarse))

    image, depth, weights = render_image_depth(
        rgb, sigma, points)

    assert image.shape == (2, 100, 100, 3)
    assert depth.shape == (2, 100, 100)
    assert weights.shape == (2, 100, 100, n_coarse)


def test_positional_encoding(n_coarse, pos_embedding_xyz):
    # Generate random rays
    rays = tf.random.uniform(shape=(2, 100, 100, n_coarse, 3))

    # Positional encode the rays
    pos_encoded_rays = positional_encoding(rays, pos_embedding_xyz)

    assert pos_encoded_rays.shape == (
        2, 100, 100, n_coarse, 3 * 2 * pos_embedding_xyz + 3)


def test_fine_hierarchical_sampling(n_coarse, n_fine):
    n_fine = 64
    # Generate random mid_points and weights
    coarse_points = tf.random.uniform(shape=(2, 100, 100, n_coarse))
    mid_points = 0.5 * (coarse_points[..., 1:] + coarse_points[..., :-1])

    weights = tf.random.uniform(shape=(2, 100, 100, n_coarse))

    # Apply hierarchical sampling
    fine_points = fine_hierarchical_sampling(
        mid_points, weights, n_fine)

    assert fine_points.shape == (2, 100, 100, n_fine)


def test_encode_position_and_directions(n_coarse, pos_embedding_xyz, pos_embedding_dir):
    ray_origin = tf.random.uniform(shape=(2, 100, 100, 3))
    ray_direction = tf.random.uniform(shape=(2, 100, 100, 3))
    sample_points = tf.random.uniform(shape=(2, 100, 100, n_coarse))

    # Encode the position and directions
    pos_encoded_rays, pos_encoded_directions = encode_position_and_directions(
        ray_origin, ray_direction, sample_points, pos_embedding_xyz, pos_embedding_dir)

    assert pos_encoded_rays.shape == (
        2, 100, 100, n_coarse, 3 * 2 * pos_embedding_xyz + 3)
    assert pos_encoded_directions.shape == (
        2, 100, 100, n_coarse, 3 * 2 * pos_embedding_dir + 3)
