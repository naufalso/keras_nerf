import logging
import os
import tensorflow as tf
import json

from keras_nerf.model.nerf.mlp import NeRFMLP
from keras_nerf.model.nerf.utils import NeRFUtils


class NeRF(tf.keras.Model):
    def __init__(self, n_coarse: int = 64, n_fine: int = 128,
                 pos_emb_xyz: int = 10, pos_emb_dir: int = 4,
                 n_layers: int = 8, dense_units: int = 256,
                 skip_layer=4, model_path: str = None,  **kwargs):
        super(NeRF, self).__init__(**kwargs)
        logging.info('Initializing NeRF model')
        logging.info(
            f'NeRF Parameters: n_coarse={n_coarse}, n_fine={n_fine}, '
            f'pos_emb_xyz={pos_emb_xyz}, pos_emb_dir={pos_emb_dir}, '
            f'n_layers={n_layers}, dense_units={dense_units}')

        self.model_path = model_path

        if self.model_path is None:
            self.n_coarse = n_coarse
            self.n_fine = n_fine
            self.pos_emb_xyz = pos_emb_xyz
            self.pos_emb_dir = pos_emb_dir
            self.n_layers = n_layers
            self.dense_units = dense_units
            self.skip_layer = skip_layer
            logging.info('Creating NeRF model')
        else:
            logging.info('Loading NeRF model')
            self.load_model(model_path)

        self.coarse = NeRFMLP(
            n_layers=n_layers, dense_units=dense_units, skip_layer=skip_layer, name='coarse_nerf')
        self.fine = NeRFMLP(
            n_layers=n_layers, dense_units=dense_units, skip_layer=skip_layer, name='fine_nerf')

        self.epsilon = 1e-10
        self.zero_int = tf.constant(0, dtype=tf.int64)

    def save_model(self, path, weights_only=False):
        logging.info('Saving NeRF model')
        model_config = {
            'n_coarse': self.n_coarse,
            'n_fine': self.n_fine,
            'pos_emb_xyz': self.pos_emb_xyz,
            'pos_emb_dir': self.pos_emb_dir,
            'n_layers': self.n_layers,
            'dense_units': self.dense_units,
            'skip_layer': self.skip_layer
        }

        os.makedirs(path, exist_ok=True)

        if not weights_only:
            with open(os.path.join(path, 'model_config.json'), 'w') as f:
                json.dump(model_config, f)

        self.coarse.save_weights(os.path.join(path, 'coarse.h5'))
        self.fine.save_weights(os.path.join(path, 'fine.h5'))

    def load_model(self, path):
        with open(os.path.join(path, 'model_config.json'), 'r') as f:
            model_config = json.load(f)

        self.n_coarse = model_config['n_coarse']
        self.n_fine = model_config['n_fine']
        self.pos_emb_xyz = model_config['pos_emb_xyz']
        self.pos_emb_dir = model_config['pos_emb_dir']
        self.n_layers = model_config['n_layers']
        self.dense_units = model_config['dense_units']
        self.skip_layer = model_config['skip_layer']

    def compile(self, optimizer, loss, batch_size, image_height, image_width, ray_chunks, white_background=False, is_training=True, **kwargs):
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
        self.white_background = white_background

        self.ray_chunks = ray_chunks
        self.num_rays = batch_size * image_height * image_width

        if self.ray_chunks >= self.num_rays:
            self.ray_chunks = self.num_rays
            logging.info(
                f"ray_chunks is greater than num_rays, setting ray_chunks to num_rays: {self.num_rays}")

        assert self.num_rays % self.ray_chunks == 0, f'ray_chunks {self.ray_chunks} must be a divisor of the number of rays {self.num_rays}'

        self.sequential_chunks = self.num_rays // ray_chunks
        self.sequential_chunks_fl = tf.cast(self.sequential_chunks, tf.float32)
        logging.info(
            f'NeRF Sequential Model Prediction: num_rays={self.num_rays}, sequential_chunks={self.sequential_chunks}')

        self.nerf_utils = NeRFUtils(
            self.batch_size, self.image_height, self.image_width, self.ray_chunks, self.pos_emb_xyz, self.pos_emb_dir, self.white_background
        )

        self._build_model()
        if is_training:
            self._initialize_training_accumulator()
            self._initialize_optimizer_and_metrics(optimizer)

    def _build_model(self):
        # Build the coarse and fine models
        self.coarse((
            tf.random.uniform(
                [self.ray_chunks, self.n_coarse, 3 * 2 * self.pos_emb_xyz + 3]),
            tf.random.uniform(
                [self.ray_chunks, self.n_coarse, 3 * 2 * self.pos_emb_dir + 3]))
        )

        self.fine((
            tf.random.uniform(
                [self.ray_chunks, self.n_coarse + self.n_fine, 3 * 2 * self.pos_emb_xyz + 3]),
            tf.random.uniform(
                [self.ray_chunks, self.n_coarse + self.n_fine, 3 * 2 * self.pos_emb_dir + 3]))
        )

        if self.model_path is not None:
            logging.info('Loading NeRF model weights')
            self.coarse.load_weights(
                os.path.join(self.model_path, 'coarse.h5'))
            self.fine.load_weights(os.path.join(self.model_path, 'fine.h5'))

    def _initialize_training_accumulator(self):
        # Initialize gradients accumulators for the coarse and fine models
        self.coarse_gradients_accumulator = [
            tf.Variable(tf.zeros_like(var), trainable=False)
            for var in self.coarse.trainable_variables
        ]
        self.fine_gradients_accumulator = [
            tf.Variable(tf.zeros_like(var), trainable=False)
            for var in self.fine.trainable_variables
        ]

        if self.run_eagerly:
            logging.info('Running in Eager mode')
            self.coarse_zero_gradients = tf.Variable(
                0, dtype=tf.int64, trainable=False)
            self.fine_zero_gradients = tf.Variable(
                0, dtype=tf.int64, trainable=False)

        logging.debug(
            f'Total number of trainable variables - Coarse: {len(self.coarse.trainable_variables)}, Fine: {len(self.fine.trainable_variables)}')

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

    def _predict_and_render_chunk(self, ray_chunks, coarse_weights_chunk=None):
        ray_origin_chunk, ray_direction_chunk, coarse_points_chunk = ray_chunks

        # Collect the points and directions either for fine or coarse model
        if coarse_weights_chunk is not None:
            # For fine model, we need to sample the fine points and collect the coarse points as well
            # Compute middle points for fine sampling
            mid_points_chunk = 0.5 * \
                (coarse_points_chunk[..., 1:] + coarse_points_chunk[..., :-1])

            # Apply hierarchical sampling and get the fine samples for the fine rays
            fine_points_chunk = self.nerf_utils.fine_hierarchical_sampling_chunk(
                mid_points_chunk, coarse_weights_chunk, self.n_fine)

            # Concatenate the coarse and fine points
            points_chunk = tf.sort(
                tf.concat([coarse_points_chunk, fine_points_chunk], axis=-1), axis=-1)

            model = self.fine
        else:
            # For coarse model, we only need to use the coarse points
            points_chunk = coarse_points_chunk
            model = self.coarse

        # Positional encode the points and directions
        encoded_ray_chunks, encoded_ray_direction_chunks = self.nerf_utils.encode_position_and_directions(
            ray_origin_chunk, ray_direction_chunk, points_chunk
        )

        rgb_chunk, sigma_chunk = model(
            (encoded_ray_chunks, encoded_ray_direction_chunks))

        # Render the image and depth
        image_chunk, depth_chunk, weight_chunk = self.nerf_utils.render_image_depth_chunk(
            rgb_chunk, sigma_chunk, points_chunk
        )

        return {
            'image': image_chunk,
            'depth': depth_chunk,
            'weights': weight_chunk
        }

    def predict_and_render_chunk(self, ray_chunks):
        # Predict the coarse image and depth
        coarse_results = self._predict_and_render_chunk(
            ray_chunks)

        # Predict the fine image and depth
        fine_results = self._predict_and_render_chunk(
            ray_chunks, coarse_results['weights'])

        return coarse_results, fine_results

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
             coarse_weights_chunk) = coarse_chunk_results['image'], coarse_chunk_results['depth'], coarse_chunk_results['weights']
            (fine_image_chunk, fine_depth_chunk,
             fine_weights_chunk) = fine_chunk_results['image'], fine_chunk_results['depth'], fine_chunk_results['weights']

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

        coarse_results = {
            'image': coarse_image,
            'depth': coarse_depth,
            'weights': coarse_weights
        }

        fine_results = {
            'image': fine_image,
            'depth': fine_depth,
            'weights': fine_weights
        }

        return coarse_results, fine_results

    def update_and_return_metrics(self, images, coarse_images, fine_images, coarse_loss, fine_loss):
        # Compute the PSNR and SSIM metrics
        logging.debug('Computing metrics')
        coarse_psnr = tf.image.psnr(images, coarse_images, max_val=1.0)
        coarse_ssim = tf.image.ssim(images, coarse_images, max_val=1.0)
        fine_psnr = tf.image.psnr(images, fine_images, max_val=1.0)
        fine_ssim = tf.image.ssim(images, fine_images, max_val=1.0)

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

    def train_step(self, inputs):
        logging.debug("Training step")
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
        fine_image_flat = tf.TensorArray(
            tf.float32, size=self.sequential_chunks)

        # Compute gradients for coarse and fine model
        logging.debug('Computing gradients for coarse and fine model')
        for i in range(self.sequential_chunks):
            start_chunk = i * self.ray_chunks
            end_chunk = (i + 1) * self.ray_chunks

            # logging.debug(f'Chunk {i+1}/{self.sequential_chunks}')
            image_flat_chunk = image_flat[start_chunk:end_chunk, ...]
            ray_origin_chunk = ray_origin_flat[start_chunk:end_chunk, ...]
            ray_direction_chunk = ray_direction_flat[start_chunk:end_chunk, ...]
            coarse_points_chunk = coarse_points_flat[start_chunk:end_chunk, ...]

            with tf.GradientTape(watch_accessed_variables=False) as coarse_tape:
                # Watch the trainable variables.
                coarse_tape.watch(self.coarse.trainable_variables)

                # Predict the coarse images
                coarse_chunk_results = self._predict_and_render_chunk(
                    (ray_origin_chunk, ray_direction_chunk, coarse_points_chunk))
                (coarse_image_chunk,
                 coarse_weight_chunk) = coarse_chunk_results['image'], coarse_chunk_results['weights']

                # Compute the loss
                coarse_loss_chunk = self.loss(
                    image_flat_chunk, coarse_image_chunk)

            # Compute the coarse gradients
            coarse_gradients = coarse_tape.gradient(
                coarse_loss_chunk, self.coarse.trainable_variables)

            # Accumulate the coarse gradients
            for j, grad in enumerate(coarse_gradients):
                tf.debugging.assert_all_finite(
                    grad, f'Coarse Gradient {j} is not finite')
                self.coarse_gradients_accumulator[j].assign_add(
                    grad / self.sequential_chunks_fl)

            # Accumulate the coarse loss
            self.coarse_loss_accumulator.assign_add(
                coarse_loss_chunk / self.sequential_chunks_fl)

            with tf.GradientTape(watch_accessed_variables=False) as fine_tape:
                # Watch the trainable variables.
                fine_tape.watch(self.fine.trainable_variables)

                # Predict the fine images
                fine_chunk_results = self._predict_and_render_chunk(
                    (ray_origin_chunk, ray_direction_chunk,
                     coarse_points_chunk), coarse_weight_chunk
                )
                fine_image_chunk = fine_chunk_results['image']

                # Compute the loss
                fine_loss_chunk = self.loss(image_flat_chunk, fine_image_chunk)

            # Compute the fine gradients
            fine_gradients = fine_tape.gradient(
                fine_loss_chunk, self.fine.trainable_variables)

            # Accumulate the fine gradients
            for j, grad in enumerate(fine_gradients):
                tf.debugging.assert_all_finite(
                    grad, f'Fine Gradient {j} is not finite')
                self.fine_gradients_accumulator[j].assign_add(
                    grad / self.sequential_chunks_fl)

            # Accumulate the fine loss
            self.fine_loss_accumulator.assign_add(
                fine_loss_chunk / self.sequential_chunks_fl)

            # Append Image Chunks
            coarse_image_flat = coarse_image_flat.write(i, coarse_image_chunk)
            fine_image_flat = fine_image_flat.write(i, fine_image_chunk)

        # Reconstruct image from chunks
        coarse_images = tf.reshape(
            coarse_image_flat.stack(), (self.batch_size, self.image_height, self.image_width, 3))
        fine_images = tf.reshape(
            fine_image_flat.stack(), (self.batch_size, self.image_height, self.image_width, 3))

        # Check if gradient is zeros
        if self.run_eagerly:
            self.coarse_zero_gradients.assign(0)
            self.fine_zero_gradients.assign(0)

            for j, grad in enumerate(coarse_gradients):
                self.coarse_zero_gradients.assign_add(
                    tf.math.count_nonzero(grad))
            for j, grad in enumerate(fine_gradients):
                self.fine_zero_gradients.assign_add(
                    tf.math.count_nonzero(grad))

            if self.coarse_zero_gradients == 0 and self.fine_zero_gradients == 0:
                logging.error(
                    f'Both Coarse and Fine Gradient are zero')

            elif self.coarse_zero_gradients == 0:
                logging.warning(
                    f'Coarse Gradient is zero')

            elif self.fine_zero_gradients == 0:
                logging.warning(
                    f'Fine Gradient is zero')

        # Update the model weights using backpropagation
        logging.debug('Updating model weights')
        self.coarse_optimizer.apply_gradients(
            zip(self.coarse_gradients_accumulator, self.coarse.trainable_variables))
        self.fine_optimizer.apply_gradients(
            zip(self.fine_gradients_accumulator, self.fine.trainable_variables))

        # Calculate and update metrics
        metrices = self.update_and_return_metrics(
            images, coarse_images, fine_images, self.coarse_loss_accumulator, self.fine_loss_accumulator)

        # Reset the accumulators to zero
        self.coarse_loss_accumulator.assign(0.0)
        self.fine_loss_accumulator.assign(0.0)

        for var in self.coarse_gradients_accumulator:
            var.assign(tf.zeros_like(var))
        for var in self.fine_gradients_accumulator:
            var.assign(tf.zeros_like(var))

        return metrices

    def test_step(self, inputs):
        logging.debug('Testing step')
        # Unpack the data.
        images, rays = inputs
        images = images[..., :3]

        # Predict the coarse and fine images
        logging.debug('Predicting coarse and fine images')
        coarse_results, fine_results = self.predict_and_render_images(rays)
        coarse_images = coarse_results['image']
        fine_images = fine_results['image']

        # Compute the photometric loss for the coarse model
        logging.debug('Calculating coarse loss')
        coarse_loss = self.loss(images, coarse_images)

        # Compute the photometric loss for fine model
        logging.debug('Calculating fine loss')
        fine_loss = self.loss(images, fine_images)

        # Update the loss and metrics trackers
        return self.update_and_return_metrics(
            images, coarse_images, fine_images, coarse_loss, fine_loss)

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
