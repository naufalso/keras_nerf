import pytest
import tensorflow as tf

from keras_nerf.data.image import ImageLoader


@pytest.fixture
def image_loader():
    return ImageLoader(128, 128)


def test_image_loader_output(image_loader):
    for subset in ["train", "val", "test"]:
        for i in range(10):
            image = image_loader(
                f"data/nerf_synthetic/lego/{subset}/r_{i}.png")
            assert image.shape == (128, 128, 4)
            assert image.dtype == tf.float32
            assert tf.reduce_min(image) >= 0.0
            assert tf.reduce_max(image) <= 1.0
