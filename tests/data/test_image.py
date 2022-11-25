import pytest
import tensorflow as tf

from keras_nerf.data.image import ImageLoader


@pytest.fixture
def image_loader():
    return ImageLoader(100, 100)


def test_image_loader_output(image_loader):
    image = image_loader(
        "data/nerf_synthetic/lego/train/r_0.png")
    assert image.shape == (100, 100, 4)
    assert image.dtype == tf.float32
    assert tf.reduce_min(image) >= 0.0
    assert tf.reduce_max(image) <= 1.0
