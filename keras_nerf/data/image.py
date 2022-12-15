import tensorflow as tf


class ImageLoader:
    def __init__(self, image_width: int, image_height: int, white_background: bool = False, **kwargs):
        """
        Initialize the image loader.
        Args:
            image_width (int): width of the image.
            image_height (int): height of the image.
        """

        self.image_width = image_width
        self.image_height = image_height
        self.white_background = white_background

    @tf.function
    def __call__(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=4, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(
            image, (self.image_width, self.image_height), antialias=True)

        if self.white_background:
            background = tf.ones_like(image[..., :3])
        else:
            background = tf.zeros_like(image[..., :3])
        alpha = tf.expand_dims(image[:, :, -1], axis=-1)

        image = alpha * image[..., :3] + (1.0 - alpha) * background
        image = tf.concat([image, alpha], axis=-1)
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image
