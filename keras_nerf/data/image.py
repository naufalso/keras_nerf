import tensorflow as tf


class ImageLoader:
    def __init__(self, image_width: int, image_height: int, **kwargs):
        """
        Initialize the image loader.
        Args:
            image_width (int): width of the image.
            image_height (int): height of the image.
        """

        self.image_width = image_width
        self.image_height = image_height

    @tf.function
    def __call__(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=4, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(
            image, (self.image_width, self.image_height), antialias=True)
        return image
