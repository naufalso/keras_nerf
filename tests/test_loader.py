import pytest
import tensorflow as tf
import os

from keras_nerf.data.loader import DatasetLoader


@pytest.fixture
def dataset_loader():
    return DatasetLoader("data/nerf_synthetic/lego")


def test_load_image_camera_list(dataset_loader):
    json_config = dataset_loader._load_json(
        os.path.join(dataset_loader.data_dir, "transforms_train.json")
    )

    assert isinstance(json_config, dict)

    image_paths, camera_params = dataset_loader._load_image_path_and_camera_param(
        json_config)

    assert len(image_paths) == 100
    assert len(camera_params) == 100


def test_dataset_loader(dataset_loader):
    train_ds, val_ds, test_ds = dataset_loader.load_dataset(
        batch_size=2,
        image_width=200,
        image_height=200,
        near=2.0,
        far=6.0,
        n_sample=32
    )

    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(val_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)

    for ds in [train_ds, val_ds, test_ds]:
        for batch in ds.take(1):
            images, rays = batch
            ray_origin, ray_direction, sample_points = rays

            assert ray_origin.shape == (2, 200, 200, 3)
            assert ray_direction.shape == (2, 200, 200, 3)
            assert sample_points.shape == (2, 200, 200, 32)
            assert images.shape == (2, 200, 200, 4)
