import logging
import os
import tensorflow as tf

from keras_nerf.model.nerf.mlp import NeRFMLP
from keras_nerf.model.nerf.utils import NeRFUtils


class NeRF(tf.keras.Model):
    def __init__(self, n_coarse: int = 64, n_fine: int = 128,
                 pos_emb_xyz: int = 10, pos_emb_dir: int = 4,
                 n_layers: int = 8, dense_units: int = 256,
                 skip_layer=4, model_path: str = None,  **kwargs):
        super(NeRF, self).__init__(**kwargs)
        logging.info('Building NeRF model')
        logging.info(
            f'NeRF Parameters: n_coarse={n_coarse}, n_fine={n_fine}, '
            f'pos_emb_xyz={pos_emb_xyz}, pos_emb_dir={pos_emb_dir}, '
            f'n_layers={n_layers}, dense_units={dense_units}')

        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.pos_emb_xyz = pos_emb_xyz
        self.pos_emb_dir = pos_emb_dir
        self.n_layers = n_layers
        self.dense_units = dense_units
        self.skip_layer = skip_layer
        self.model_path = model_path

        if model_path is None:
            self.coarse = NeRFMLP(
                n_layers=n_layers, dense_units=dense_units, skip_layer=skip_layer, name='coarse_nerf')
            self.fine = NeRFMLP(
                n_layers=n_layers, dense_units=dense_units, skip_layer=skip_layer, name='fine_nerf')
        else:
            self.coarse = tf.keras.models.load_model(
                os.path.join(model_path, f"coarse"), compile=False)
            self.fine = tf.keras.models.load_model(
                os.path.join(model_path, f"fine"), compile=False)
        self.epsilon = 1e-10

    def compile(self, optimizer, loss, batch_size, image_height, image_width, ray_chunks, **kwargs):
        super(NeRF, self).compile(**kwargs)
        logging.info('Compiling NeRF model')
        logging.info(
            f'NeRF Compalation Parameters:'
            f'batch_size={batch_size}, image_height={image_height}, image_width={image_width}, ray_chunks={ray_chunks}'
        )
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width

        self.ray_chunks = ray_chunks
        self.num_rays = batch_size * image_height * image_width

        if self.ray_chunks >= self.num_rays:
            self.ray_chunks = self.num_rays
            logging.info(
                f"ray_chunks is greater than num_rays, setting ray_chunks to num_rays: {self.num_rays}")

        assert self.num_rays % self.ray_chunks == 0, 'ray_chunks must be a divisor of the number of rays'

        self.sequential_chunks = self.num_rays // ray_chunks
        logging.info(
            f'NeRF Sequential Model Prediction: num_rays={self.num_rays}, sequential_chunks={self.sequential_chunks}')

        self.nerf_utils = NeRFUtils(
            self.batch_size, self.image_height, self.image_width, self.ray_chunks
        )
        self._build_model_and_trainable_variables()
        self._initialize_optimizer_and_metrics(optimizer)
        self._initialize_last_prediction_samples()

    def _build_model_and_trainable_variables(self):
        # Build the coarse and fine models
        self.coarse.build(input_shape=[
            [self.ray_chunks, self.n_coarse, 3 * 2 * self.pos_emb_xyz + 3],
            [self.ray_chunks, self.n_coarse, 3 * 2 * self.pos_emb_dir + 3]]
        )

        self.fine.build(input_shape=[
            [self.ray_chunks, self.n_coarse + self.n_fine,
                3 * 2 * self.pos_emb_xyz + 3],
            [self.ray_chunks, self.n_coarse + self.n_fine, 3 * 2 * self.pos_emb_dir + 3]]
        )

        # Initialize gradients accumulators for the coarse and fine models
        self.coarse_gradients_accumulator = [
            tf.Variable(tf.zeros_like(var), trainable=False)
            for var in self.coarse.trainable_variables
        ]
        self.fine_gradients_accumulator = [
            tf.Variable(tf.zeros_like(var), trainable=False)
            for var in self.fine.trainable_variables
        ]

        # Initialize the loss accumulator
        self.coarse_loss_accumulator = tf.Variable(0.0, trainable=False)
        self.fine_loss_accumulator = tf.Variable(0.0, trainable=False)

    def _initialize_optimizer_and_metrics(self, optimizer):
        self.coarse_optimizer = tf.keras.optimizers.get(optimizer)
        self.fine_optimizer = tf.keras.optimizers.get(optimizer)

        self.coarse_loss_tracker = tf.keras.metrics.Mean(name="coarse_loss")
        self.coarse_psnr_metric = tf.keras.metrics.Mean(name="coarse_psnr")
        self.corase_ssim_metric = tf.keras.metrics.Mean(name="coarse_ssim")

        self.fine_loss_tracker = tf.keras.metrics.Mean(name="fine_loss")
        self.fine_psnr_metric = tf.keras.metrics.Mean(name="fine_psnr")
        self.fine_ssim_metric = tf.keras.metrics.Mean(name="fine_ssim")

    def _initialize_last_prediction_samples(self):
        self.last_train_coarse_image = tf.Variable(
            tf.zeros((self.batch_size, self.image_height, self.image_width, 3)))
        self.last_train_fine_image = tf.Variable(
            tf.zeros((self.batch_size, self.image_height, self.image_width, 3)))
        self.last_train_coarse_depth = tf.Variable(
            tf.zeros((self.batch_size, self.image_height, self.image_width)))
        self.last_train_fine_depth = tf.Variable(
            tf.zeros((self.batch_size, self.image_height, self.image_width)))
        self.last_train_image = tf.Variable(
            tf.zeros((self.batch_size, self.image_height, self.image_width, 3)))

    def predict_and_render_chunk(self, ray_chunks):
        ray_origin_chunk, ray_direction_chunk, coarse_points_chunk = ray_chunks
        # Encode coarse rays
        coarse_rays_chunk, coarse_rays_direction_chunk = self.nerf_utils.encode_position_and_directions(
            ray_origin_chunk, ray_direction_chunk, coarse_points_chunk, self.pos_emb_xyz, self.pos_emb_dir)

        coarse_rays_chunk = tf.ensure_shape(
            coarse_rays_chunk, [self.ray_chunks, self.n_coarse, 3 * 2 * self.pos_emb_xyz + 3])
        coarse_rays_direction_chunk = tf.ensure_shape(
            coarse_rays_direction_chunk, [self.ray_chunks, self.n_coarse, 3 * 2 * self.pos_emb_dir + 3])

        # Predict coarse rays
        coarse_rgb_chunk, coarse_sigma_chunk = self.coarse(
            (coarse_rays_chunk, coarse_rays_direction_chunk))

        # Render coarse image and depth
        coarse_image_chunk, coarse_depth_chunk, coarse_weights_chunk = self.nerf_utils.render_image_depth_chunk(
            coarse_rgb_chunk, coarse_sigma_chunk, coarse_points_chunk
        )

        # Compute middle points for fine sampling
        mid_points_chunk = 0.5 * \
            (coarse_points_chunk[..., 1:] + coarse_points_chunk[..., :-1])

        # Apply hierarchical sampling and get the fine samples for the fine rays
        fine_points_chunk = self.nerf_utils.fine_hierarchical_sampling_chunk(
            mid_points_chunk, coarse_weights_chunk, self.n_fine)

        # Combine the coarse and fine points
        fine_points_chunk = tf.sort(
            tf.concat([coarse_points_chunk, fine_points_chunk], axis=-1), axis=-1)

        # Encode the fine rays
        fine_rays_chunk, fine_rays_direction_chunk = self.nerf_utils.encode_position_and_directions(
            ray_origin_chunk, ray_direction_chunk, fine_points_chunk,  self.pos_emb_xyz, self.pos_emb_dir)

        # Compute the fine rgb and sigma
        fine_rgb_chunk, fine_sigma_chunk = self.fine(
            (fine_rays_chunk, fine_rays_direction_chunk))

        # Render the fine image and depth
        fine_image_chunk, fine_depth_chunk, fine_weights_chunk = self.nerf_utils.render_image_depth_chunk(
            fine_rgb_chunk, fine_sigma_chunk, fine_points_chunk)

        return (coarse_image_chunk, coarse_depth_chunk, coarse_weights_chunk), (fine_image_chunk, fine_depth_chunk, fine_weights_chunk)

    def predict_and_render_images(self, rays):
        # Unpack the data.
        ray_origin, ray_direction, coarse_points = rays

        ray_origin_flat = tf.reshape(ray_origin, (self.num_rays, 3))
        ray_direction_flat = tf.reshape(ray_direction, (self.num_rays, 3))
        coarse_points_flat = tf.reshape(
            coarse_points, (self.num_rays, self.n_coarse))

        coarse_image_flat = tf.TensorArray(
            tf.float32, size=self.sequential_chunks)
        coarse_depth_flat = tf.TensorArray(
            tf.float32, size=self.sequential_chunks)
        coarse_weight_flat = tf.TensorArray(
            tf.float32, size=self.sequential_chunks)
        fine_image_flat = tf.TensorArray(
            tf.float32, size=self.sequential_chunks)
        fine_depth_flat = tf.TensorArray(
            tf.float32, size=self.sequential_chunks)
        fine_weight_flat = tf.TensorArray(
            tf.float32, size=self.sequential_chunks)

        for i in range(self.sequential_chunks):
            start_chunk = i * self.ray_chunks
            end_chunk = (i + 1) * self.ray_chunks

            ray_origin_chunk = ray_origin_flat[start_chunk:end_chunk, ...]
            ray_direction_chunk = ray_direction_flat[start_chunk:end_chunk, ...]
            coarse_points_chunk = coarse_points_flat[start_chunk:end_chunk, ...]

            coarse_chunk_results, fine_chunk_results = self.predict_and_render_chunk(
                (ray_origin_chunk, ray_direction_chunk, coarse_points_chunk))

            (coarse_image_chunk, coarse_depth_chunk,
             coarse_weights_chunk) = coarse_chunk_results
            (fine_image_chunk, fine_depth_chunk,
             fine_weights_chunk) = fine_chunk_results

            # Write the results to the tensor array
            coarse_image_flat = coarse_image_flat.write(i, coarse_image_chunk)
            coarse_depth_flat = coarse_depth_flat.write(i, coarse_depth_chunk)
            coarse_weight_flat = coarse_weight_flat.write(
                i, coarse_weights_chunk)

            fine_image_flat = fine_image_flat.write(i, fine_image_chunk)
            fine_depth_flat = fine_depth_flat.write(i, fine_depth_chunk)
            fine_weight_flat = fine_weight_flat.write(i, fine_weights_chunk)

        # Stack the results
        coarse_image = tf.reshape(
            coarse_image_flat.stack(), (self.batch_size, self.image_height, self.image_width, 3))
        coarse_depth = tf.reshape(
            coarse_depth_flat.stack(), (self.batch_size, self.image_height, self.image_width))
        coarse_weights = tf.reshape(
            coarse_weight_flat.stack(), (self.batch_size, self.image_height, self.image_width, self.n_coarse))

        fine_image = tf.reshape(
            fine_image_flat.stack(), (self.batch_size, self.image_height, self.image_width, 3))
        fine_depth = tf.reshape(
            fine_depth_flat.stack(), (self.batch_size, self.image_height, self.image_width))
        fine_weights = tf.reshape(
            fine_weight_flat.stack(), (self.batch_size, self.image_height, self.image_width, self.n_coarse + self.n_fine))

        return (coarse_image, coarse_depth, coarse_weights), (fine_image, fine_depth, fine_weights)

    def train_step(self, inputs):
        images, rays = inputs
        images = images[..., :3]
        ray_origin, ray_direction, coarse_points = rays

        image_flat = tf.reshape(images, (self.num_rays, 3))
        ray_origin_flat = tf.reshape(ray_origin, (self.num_rays, 3))
        ray_direction_flat = tf.reshape(ray_direction, (self.num_rays, 3))
        coarse_points_flat = tf.reshape(
            coarse_points, (self.num_rays, self.n_coarse))

        coarse_image_flat = tf.TensorArray(
            tf.float32, size=self.sequential_chunks)
        coarse_depth_flat = tf.TensorArray(
            tf.float32, size=self.sequential_chunks)

        fine_image_flat = tf.TensorArray(
            tf.float32, size=self.sequential_chunks)
        fine_depth_flat = tf.TensorArray(
            tf.float32, size=self.sequential_chunks)

        # Compute gradients for coarse and fine model
        logging.debug('Computing gradients for coarse and fine model')
        for i in range(self.sequential_chunks):
            start_chunk = i * self.ray_chunks
            end_chunk = (i + 1) * self.ray_chunks

            image_flat_chunk = image_flat[start_chunk:end_chunk, ...]
            ray_origin_chunk = ray_origin_flat[start_chunk:end_chunk, ...]
            ray_direction_chunk = ray_direction_flat[start_chunk:end_chunk, ...]
            coarse_points_chunk = coarse_points_flat[start_chunk:end_chunk, ...]

            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                # Watch the trainable variables.
                tape.watch(self.coarse.trainable_variables)
                tape.watch(self.fine.trainable_variables)

                # Predict the coarse and fine images
                coarse_chunk_results, fine_chunk_results = self.predict_and_render_chunk(
                    (ray_origin_chunk, ray_direction_chunk, coarse_points_chunk))

                (coarse_image_chunk, coarse_depth_chunk, _) = coarse_chunk_results
                (fine_image_chunk, fine_depth_chunk, _) = fine_chunk_results

                # Compute the loss
                coarse_loss_chunk = self.loss(
                    image_flat_chunk, coarse_image_chunk)
                fine_loss_chunk = self.loss(image_flat_chunk, fine_image_chunk)

                self.coarse_loss_accumulator.assign_add(
                    coarse_loss_chunk / self.sequential_chunks)
                self.fine_loss_accumulator.assign_add(
                    fine_loss_chunk / self.sequential_chunks)

            # Append Image Chunks
            coarse_image_flat = coarse_image_flat.write(i, coarse_image_chunk)
            coarse_depth_flat = coarse_depth_flat.write(i, coarse_depth_chunk)
            fine_image_flat = fine_image_flat.write(i, fine_image_chunk)
            fine_depth_flat = fine_depth_flat.write(i, fine_depth_chunk)

            # Compute the gradients
            coarse_gradients = tape.gradient(
                coarse_loss_chunk, self.coarse.trainable_variables)
            fine_gradients = tape.gradient(
                fine_loss_chunk, self.fine.trainable_variables)

            # Accumulate the gradients
            for j, grad in enumerate(coarse_gradients):
                self.coarse_gradients_accumulator[j].assign_add(
                    grad / self.sequential_chunks)
            for j, grad in enumerate(fine_gradients):
                self.fine_gradients_accumulator[j].assign_add(
                    grad / self.sequential_chunks)

        # Update the model weights using backpropagation
        logging.debug('Updating model weights')
        self.coarse_optimizer.apply_gradients(
            zip(self.coarse_gradients_accumulator, self.coarse.trainable_variables))
        self.fine_optimizer.apply_gradients(
            zip(self.fine_gradients_accumulator, self.fine.trainable_variables))

        # Reconstruct image from chunks
        coarse_image = tf.reshape(
            coarse_image_flat.stack(), (self.batch_size, self.image_height, self.image_width, 3))
        coarse_depth = tf.reshape(
            coarse_depth_flat.stack(), (self.batch_size, self.image_height, self.image_width))

        fine_image = tf.reshape(
            fine_image_flat.stack(), (self.batch_size, self.image_height, self.image_width, 3))
        fine_depth = tf.reshape(
            fine_depth_flat.stack(), (self.batch_size, self.image_height, self.image_width))

        # Compute the PSNR and SSIM metrics
        logging.debug('Computing metrics')
        coarse_psnr = tf.image.psnr(images, coarse_image, max_val=1.0)
        coarse_ssim = tf.image.ssim(images, coarse_image, max_val=1.0)
        fine_psnr = tf.image.psnr(images, fine_image, max_val=1.0)
        fine_ssim = tf.image.ssim(images, fine_image, max_val=1.0)

        # Update the loss and metrics trackers
        logging.debug('Updating loss and metrics trackers')
        self.coarse_loss_tracker.update_state(self.coarse_loss_accumulator)
        self.coarse_psnr_metric.update_state(coarse_psnr)
        self.corase_ssim_metric.update_state(coarse_ssim)
        self.fine_loss_tracker.update_state(self.fine_loss_accumulator)
        self.fine_psnr_metric.update_state(fine_psnr)
        self.fine_ssim_metric.update_state(fine_ssim)

        # Reset the accumulators to zero
        self.coarse_loss_accumulator.assign(0.0)
        self.fine_loss_accumulator.assign(0.0)

        for var in self.coarse_gradients_accumulator:
            var.assign(tf.zeros_like(var))
        for var in self.fine_gradients_accumulator:
            var.assign(tf.zeros_like(var))

        # Save last rendered images
        self.last_train_coarse_image.assign(coarse_image)
        self.last_train_fine_image.assign(fine_image)
        self.last_train_coarse_depth.assign(coarse_depth)
        self.last_train_fine_depth.assign(fine_depth)
        self.last_train_image.assign(images)

        return {
            "coarse_loss": self.coarse_loss_tracker.result(),
            "coarse_psnr": self.coarse_psnr_metric.result(),
            "coarse_ssim": self.corase_ssim_metric.result(),
            "fine_loss": self.fine_loss_tracker.result(),
            "fine_psnr": self.fine_psnr_metric.result(),
            "fine_ssim": self.fine_ssim_metric.result(),
        }

    def test_step(self, inputs):
        logging.debug('Testing step')
        # Unpack the data.
        images, rays = inputs
        images = images[..., :3]

        # Predict the coarse and fine images
        logging.debug('Predicting coarse and fine images')
        coarse_results, fine_results = self.predict_and_render_images(rays)
        (coarse_image, _, _) = coarse_results
        (fine_image, _, _) = fine_results

        # Compute the photometric loss for the coarse model
        logging.debug('Calculating coarse loss')
        coarse_loss = self.loss(images, coarse_image)

        # Compute the photometric loss for fine model
        logging.debug('Calculating fine loss')
        fine_loss = self.loss(images, fine_image)

        # Compute the PSNR and SSIM metrics
        logging.debug('Computing metrics')
        coarse_psnr = tf.image.psnr(images, coarse_image, max_val=1.0)
        coarse_ssim = tf.image.ssim(images, coarse_image, max_val=1.0)
        fine_psnr = tf.image.psnr(images, fine_image, max_val=1.0)
        fine_ssim = tf.image.ssim(images, fine_image, max_val=1.0)

        # Update the loss and metrics trackers
        logging.debug('Updating loss and metrics trackers')
        self.coarse_loss_tracker.update_state(coarse_loss)
        self.coarse_psnr_metric.update_state(coarse_psnr)
        self.corase_ssim_metric.update_state(coarse_ssim)
        self.fine_loss_tracker.update_state(fine_loss)
        self.fine_psnr_metric.update_state(fine_psnr)
        self.fine_ssim_metric.update_state(fine_ssim)

        return {
            "coarse_loss": self.coarse_loss_tracker.result(),
            "coarse_psnr": self.coarse_psnr_metric.result(),
            "coarse_ssim": self.corase_ssim_metric.result(),
            "fine_loss": self.fine_loss_tracker.result(),
            "fine_psnr": self.fine_psnr_metric.result(),
            "fine_ssim": self.fine_ssim_metric.result(),
        }

    @property
    def metrics(self):
        return [
            self.coarse_loss_tracker,
            self.coarse_psnr_metric,
            self.corase_ssim_metric,
            self.fine_loss_tracker,
            self.fine_psnr_metric,
            self.fine_ssim_metric,
        ]
