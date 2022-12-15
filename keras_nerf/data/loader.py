import os
import json
import logging

import tensorflow as tf

from keras_nerf.data.utils import *
from keras_nerf.data.rays import RaysGenerator
from keras_nerf.data.image import ImageLoader


class DatasetLoader:
    def __init__(self, data_dir: str, white_background: bool = False, **kwargs):
        """
        Initialize the dataset loader.
        Args:
            data_dir (str): path to the data directory.
        """

        self.data_dir = data_dir
        self.white_background = white_background

    def _load_json(self, filename: str) -> dict:
        """
        Load a json file.
        Args:
            filename (str): path to the json file.
        Returns:
            dict: the loaded json file.
        """

        with open(filename, 'r') as f:
            return json.load(f)

    def _load_image_path_and_camera_param(self, json_config: dict) -> tuple:
        """
        Load an image path and its camera parameters.
        Args:
            json_config (dict): the json config file.
        Returns:
            tuple: the image path and its camera parameters (camera2world matrics).
        """

        image_paths = []
        camera_params = []

        for frame in json_config['frames']:
            image_path = os.path.join(
                self.data_dir, f"{frame['file_path']}.png")
            image_paths.append(image_path)
            camera_params.append(frame['transform_matrix'])

        return image_paths, camera_params

    def load_dataset(self, batch_size: int, image_width: int, image_height: int, near: float, far: float, n_sample: int) -> tf.data.Dataset:
        """
        Load a dataset.
        Args:
            batch_size (int): batch size.
            image_width (int): image width.
            image_height (int): image height.
            near (float): near plane.
            far (float): far plane.
            n_sample (int): number of samples.
        Returns:
            tf.data.Dataset: the loaded dataset.
        """

        image_loader = ImageLoader(
            image_width, image_height, self.white_background)

        tf_datasets = []

        for subset in ['train', 'val', 'test']:
            json_config = self._load_json(
                os.path.join(self.data_dir, f"transforms_{subset}.json"))

            focal_length = get_focal_from_fov(
                json_config['camera_angle_x'], image_width)

            rays_generator = RaysGenerator(
                focal_length=focal_length,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                n_sample=n_sample)

            image_paths, camera_params = self._load_image_path_and_camera_param(
                json_config)

            tf_ds_images = tf.data.Dataset.from_tensor_slices(image_paths).map(
                image_loader
            )

            tf_ds_rays = tf.data.Dataset.from_tensor_slices(camera_params).map(
                rays_generator
            )

            tf_dataset = tf.data.Dataset.zip((tf_ds_images, tf_ds_rays))
            tf_dataset = (
                tf_dataset
                # .repeat()
                .shuffle(batch_size)
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )

            tf_datasets.append(tf_dataset)
            logging.info(
                f"Loaded {subset} dataset. {len(image_paths)} images.")

        return tf_datasets
