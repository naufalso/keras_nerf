import tensorflow as tf


class RaysGenerator:
    def __init__(self, focal_length: float, image_width: int, image_height: int, near: float, far: float, n_sample: int, **kwargs):
        """
        Initialize the rays generator.
        Args:
            focal_length (float): focal length of the camera.
            image_width (int): width of the image.
            image_height (int): height of the image.
            near (float): near plane of the camera.
            far (float): far plane of the camera.
            n_sample (int): number of samples.
        """

        self.focal_length = tf.constant(focal_length, dtype=tf.float32)
        self.image_width = tf.constant(image_width, dtype=tf.int32)
        self.image_height = tf.constant(image_height, dtype=tf.int32)
        self.near = tf.constant(near, dtype=tf.float32)
        self.far = tf.constant(far, dtype=tf.float32)
        self.n_sample = tf.constant(n_sample, dtype=tf.int32)

        self.image_width_fl = tf.cast(image_width, dtype=tf.float32)
        self.image_height_fl = tf.cast(image_height, dtype=tf.float32)
        self.n_sample_fl = tf.cast(n_sample, dtype=tf.float32)
        self.focal_length_fl = tf.cast(focal_length, dtype=tf.float32)

    # def __call__(self, camera2world):
    #     # create a meshgrid of image dimensions
    #     (x, y) = tf.meshgrid(
    #         tf.range(self.image_width, dtype=tf.float32),
    #         tf.range(self.image_height, dtype=tf.float32),
    #         indexing="xy",
    #     )
    #     # define the camera coordinates
    #     xCamera = (x - self.image_width_fl * 0.5) / self.focal_length
    #     yCamera = (y - self.image_height_fl * 0.5) / self.focal_length
    #     # define the camera vector
    #     xCyCzC = tf.stack([xCamera, -yCamera, -tf.ones_like(x)],
    #                       axis=-1)
    #     # slice the camera2world matrix to obtain the rotation and
    #     # translation matrix
    #     rotation = camera2world[:3, :3]
    #     translation = camera2world[:3, -1]

    #     # expand the camera coordinates to
    #     xCyCzC = xCyCzC[..., None, :]

    #     # get the world coordinates
    #     xWyWzW = xCyCzC * rotation

    #     # calculate the direction vector of the ray
    #     rayD = tf.reduce_sum(xWyWzW, axis=-1)
    #     rayD = rayD / tf.norm(rayD, axis=-1, keepdims=True)
    #     # calculate the origin vector of the ray
    #     rayO = tf.broadcast_to(translation, tf.shape(rayD))
    #     # get the sample points from the ray
    #     tVals = tf.linspace(self.near, self.far, self.n_sample)

    #     noiseShape = list(rayO.shape[:-1]) + [self.n_sample]

    #     noise = (tf.random.uniform(shape=noiseShape, seed=0) *
    #              (self.far - self.near) / self.n_sample_fl)
    #     tVals = tVals + noise
    #     # return ray origin, direction, and the sample points
    #     return (rayO, rayD, tVals)

    @tf.function(reduce_retracing=True)
    def __call__(self, camera_params):
        """
        Generate rays.
        Vector ray = vector origin of the ray + t * vector direction of the ray

        Args:
            camera_params (tf.Tensor): camera parameters (camera2world matrics).
        Returns:
            tuple: ray origin, direction, and the sample points
        """

        # Create a meshgrid of image dimensions
        x, y = tf.meshgrid(
            tf.range(self.image_width, dtype=tf.float32),
            tf.range(self.image_height, dtype=tf.float32),
            indexing='xy'
        )  # Shape: [H, W], [H, W]

        # Define camera coordinates
        x_camera = (x - self.image_width_fl * 0.5) / self.focal_length_fl
        y_camera = (y - self.image_height_fl * 0.5) / self.focal_length_fl

        # Define camera vector (x,y,z). Shape: [H, W, 3]
        xyz_camera = tf.stack(
            [x_camera, -y_camera, -tf.ones_like(x_camera)], axis=-1)

        # Get camera to world matrix from camera parameters
        # to obtain rotation and translation matrix

        rotation = camera_params[:3, :3]  # Shape: [3, 3]
        translation = camera_params[:3, -1]  # Shape: [3]

        # Expand camera coordinates to get the world coordinates
        xyz_camera = xyz_camera[..., tf.newaxis, :]  # Shape: [H, W, 1, 3]
        xyz_world = xyz_camera * rotation  # Shape: [H, W, 3, 3]

        # Calculate the direction vector of the ray
        ray_direction = tf.reduce_sum(xyz_world, axis=-1)  # Shape: [H, W, 3]
        ray_direction = ray_direction / tf.norm(ray_direction, axis=-1,
                                                keepdims=True)  # Shape: [H, W, 3]

        # Calculate the origin vector of the ray
        ray_origin = tf.broadcast_to(
            translation, tf.shape(ray_direction))  # Shape: [H, W, 3]

        # Get the sample point from the ray
        sample_points = tf.linspace(
            self.near, self.far, self.n_sample)  # Shape: [N]

        # Add noise to the sample points
        interval = (self.far - self.near) / self.n_sample_fl

        noise = tf.random.uniform(
            shape=[self.image_width, self.image_height, self.n_sample]) * interval - (interval / 2)
        # tf.print(noise)

        sample_points = sample_points + noise  # Shape: [H, W, N]
        sample_points = tf.clip_by_value(sample_points, self.near, self.far)

        # Returns the ray origin, direction, and the sample points
        return (ray_origin, ray_direction, sample_points)
