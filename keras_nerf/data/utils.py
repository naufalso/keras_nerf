import tensorflow as tf


@tf.function(reduce_retracing=True)
def get_focal_from_fov(field_of_view: float, width: int):
    """
    Get focal length from field of view.

    Args:
        field_of_view: Field of view.
        width: Width of image.
    """
    width = tf.cast(width, tf.float32)

    return 0.5 * width / tf.tan(0.5 * field_of_view)
