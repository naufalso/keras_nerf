import logging
import tensorflow as tf

from keras_nerf.model.nerf.mlp import NeRFMLP, build_nerf_mlp
from keras_nerf.model.nerf.utils import render_image_depth, encode_position_and_directions, fine_hierarchical_sampling


class NeRF(tf.keras.Model):
    def __init__(self, n_coarse: int = 64, n_fine: int = 128, pos_emb_xyz: int = 10, pos_emb_dir: int = 4, n_layers: int = 8, dense_units: int = 256, skip_layer=4, **kwargs):
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

        self.coarse = NeRFMLP(
            n_layers=n_layers, dense_units=dense_units, skip_layer=skip_layer, name='coarse_nerf')
        self.fine = NeRFMLP(
            n_layers=n_layers, dense_units=dense_units, skip_layer=skip_layer, name='fine_nerf')
        # self.coarse = build_nerf_mlp(
        #     n_layers=n_layers, dense_units=dense_units, skip_layer=skip_layer, pos_emb_xyz=pos_emb_xyz, pos_emb_dir=pos_emb_dir, name='coarse_nerf')
        # self.fine = build_nerf_mlp(
        #     n_layers=n_layers, dense_units=dense_units, skip_layer=skip_layer, pos_emb_xyz=pos_emb_xyz, pos_emb_dir=pos_emb_dir, name='fine_nerf')
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

        assert (self.image_height *
                self.image_width) % self.ray_chunks == 0, 'ray_chunks must be a divisor of the number of rays'
        assert (self.image_height * self.image_width
                ) % self.ray_chunks == 0, 'ray_chunks must be a divisor of the number of rays'

        self.predict_coarse = self.get_sequential_model_prediction(
            tf.function(self.coarse, reduce_retracing=True), self.n_coarse)
        self.predict_fine = self.get_sequential_model_prediction(
            tf.function(self.fine, reduce_retracing=True), self.n_coarse + self.n_fine)

        self.coarse_optimizer = tf.keras.optimizers.get(optimizer)
        self.fine_optimizer = tf.keras.optimizers.get(optimizer)

        self.coarse_loss_tracker = tf.keras.metrics.Mean(name="coarse_loss")
        self.coarse_psnr_metric = tf.keras.metrics.Mean(name="coarse_psnr")
        self.corase_ssim_metric = tf.keras.metrics.Mean(name="coarse_ssim")

        self.fine_loss_tracker = tf.keras.metrics.Mean(name="fine_loss")
        self.fine_psnr_metric = tf.keras.metrics.Mean(name="fine_psnr")
        self.fine_ssim_metric = tf.keras.metrics.Mean(name="fine_ssim")

    def get_sequential_model_prediction(self, model, n_sample):
        ray_size = self.batch_size * self.image_width * self.image_height
        parallel_chunks = ray_size // self.ray_chunks

        logging.info(
            f'NeRF Sequential Model Prediction: ray_size={ray_size}, parallel_chunks={parallel_chunks}')

        def predict_ray_sequentially(rays, rays_direction):
            flat_rays = tf.reshape(
                rays, (ray_size, n_sample, 3 * 2 * self.pos_emb_xyz + 3))
            flat_rays_direction = tf.reshape(
                rays_direction, (ray_size, n_sample, 3 * 2 * self.pos_emb_dir + 3))

            rgb_array = tf.TensorArray(tf.float32, size=parallel_chunks)
            sigma_array = tf.TensorArray(tf.float32, size=parallel_chunks)

            # Split rays into chunks
            for i in range(parallel_chunks):
                start_chunk = i * self.ray_chunks
                end_chunk = (i + 1) * self.ray_chunks

                rgb, sigma = model(
                    (flat_rays[start_chunk:end_chunk, ...], flat_rays_direction[start_chunk:end_chunk, ...]))

                rgb_array = rgb_array.write(i, rgb)
                sigma_array = sigma_array.write(i, sigma)

            # Revert to original shape
            rgb = tf.reshape(
                rgb_array.stack(), (self.batch_size, self.image_width, self.image_height, n_sample, 3))
            sigma = tf.reshape(
                sigma_array.stack(), (self.batch_size, self.image_width, self.image_height, n_sample, 1))

            return rgb, sigma

        return predict_ray_sequentially

    @tf.function(reduce_retracing=True)
    def ensure_mlp_shape(self, rgb, sigma, n_points):
        rgb = tf.ensure_shape(
            rgb, (self.batch_size, self.image_width, self.image_height, n_points, 3))
        sigma = tf.ensure_shape(
            sigma, (self.batch_size, self.image_width, self.image_height, n_points, 1))
        return rgb, sigma

    @tf.function(reduce_retracing=True)
    def ensure_points_shape(self, points):
        points = tf.ensure_shape(
            points, (self.batch_size, self.image_width, self.image_height, self.n_coarse))
        return points

    @tf.function(reduce_retracing=True)
    def ensure_encoded_shape(self, encoded_ray, encoded_ray_dir, n_sample):
        encoded_ray = tf.ensure_shape(
            encoded_ray, (self.batch_size, self.image_width, self.image_height, n_sample, 3 * 2 * self.pos_emb_xyz + 3))
        encoded_ray_dir = tf.ensure_shape(
            encoded_ray_dir, (self.batch_size, self.image_width, self.image_height, n_sample, 3 * 2 * self.pos_emb_dir + 3))
        return encoded_ray, encoded_ray_dir

    def train_step(self, inputs):
        logging.debug('Training step')
        # Unpack the data.
        images, rays = inputs
        images = images[..., :3]
        ray_origin, ray_direction, coarse_points = rays

        coarse_points = self.ensure_points_shape(coarse_points)

        # Encode coarse rays
        logging.debug('Encoding coarse rays')
        coarse_rays, coarse_rays_direction = encode_position_and_directions(
            ray_origin, ray_direction, coarse_points, self.pos_emb_xyz, self.pos_emb_dir)

        coarse_rays, coarse_rays_direction = self.ensure_encoded_shape(
            coarse_rays, coarse_rays_direction, self.n_coarse)

        # Keep track of the gradients for updating coarse model
        logging.debug('Computing gradients for coarse model')
        with tf.GradientTape() as coarse_tape:
            coarse_rgb, coarse_sigma = self.predict_coarse(
                coarse_rays, coarse_rays_direction)

            # Render the coarse image and depth
            coarse_rgb, coarse_sigma = self.ensure_mlp_shape(
                coarse_rgb, coarse_sigma, self.n_coarse)

            logging.debug('Rendering coarse image')
            coarse_image, coarse_depth, coarse_weights = render_image_depth(
                coarse_rgb, coarse_sigma, coarse_points)

            # Compute the photometric loss for the coarse model
            logging.debug('Calculating coarse loss')
            coarse_loss = self.loss(images, coarse_image)

        # Compute middle points for fine sampling
        mid_points = 0.5 * (coarse_points[..., 1:] + coarse_points[..., :-1])

        # Apply hierarchical sampling and get the fine samples for the fine rays
        fine_points = fine_hierarchical_sampling(
            mid_points, coarse_weights, self.n_fine)

        # Combine the coarse and fine points
        fine_points = tf.sort(
            tf.concat([coarse_points, fine_points], axis=-1), axis=-1)

        # Encode the fine rays
        logging.debug('Encoding fine rays')
        fine_rays, fine_rays_direction = encode_position_and_directions(
            ray_origin, ray_direction, fine_points, self.pos_emb_xyz, self.pos_emb_dir)

        # Keep track of the gradients for updating fine model
        logging.debug('Computing gradients for fine model')
        with tf.GradientTape() as fine_tape:
            fine_rgb, fine_sigma = self.predict_fine(
                fine_rays, fine_rays_direction)

            # Render the fine image and depth
            fine_rgb, fine_sigma = self.ensure_mlp_shape(
                fine_rgb, fine_sigma, self.n_coarse + self.n_fine)

            logging.debug('Rendering fine image')
            fine_image, fine_depth, fine_weights = render_image_depth(
                fine_rgb, fine_sigma, fine_points)

            # Compute the photometric loss for fine model
            logging.debug('Calculating fine loss')
            fine_loss = self.loss(images, fine_image)

        # Update the model weights using backpropagation
        logging.debug('Updating model weights')
        coarse_gradients = coarse_tape.gradient(
            coarse_loss, self.coarse.trainable_variables)
        self.coarse_optimizer.apply_gradients(
            zip(coarse_gradients, self.coarse.trainable_variables))

        fine_gradients = fine_tape.gradient(
            fine_loss, self.fine.trainable_variables)
        self.fine_optimizer.apply_gradients(
            zip(fine_gradients, self.fine.trainable_variables))

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

    def test_step(self, inputs):
        logging.debug('Testing step')
        # Unpack the data.
        images, rays = inputs
        images = images[..., :3]

        ray_origin, ray_direction, coarse_points = rays
        coarse_points = self.ensure_points_shape(coarse_points)

        # Encode coarse rays
        logging.debug('Encoding coarse rays')
        coarse_rays, coarse_rays_direction = encode_position_and_directions(
            ray_origin, ray_direction, coarse_points,  self.pos_emb_xyz, self.pos_emb_dir)

        # Compute the coarse rgb and sigma
        logging.debug('Computing coarse rgb and sigma')
        coarse_rgb, coarse_sigma = self.predict_coarse(
            coarse_rays, coarse_rays_direction)

        # Render the coarse image and depth
        coarse_rgb, coarse_sigma = self.ensure_mlp_shape(
            coarse_rgb, coarse_sigma, self.n_coarse)

        logging.debug('Rendering coarse image')
        coarse_image, coarse_depth, coarse_weights = render_image_depth(
            coarse_rgb, coarse_sigma, coarse_points)

        # Compute the photometric loss for the coarse model
        logging.debug('Calculating coarse loss')
        coarse_loss = self.loss(images, coarse_image)

        # Compute middle points for fine sampling
        mid_points = 0.5 * (coarse_points[..., 1:] + coarse_points[..., :-1])

        # Apply hierarchical sampling and get the fine samples for the fine rays
        fine_points = fine_hierarchical_sampling(
            mid_points, coarse_weights, self.n_fine)

        # Combine the coarse and fine points
        fine_points = tf.sort(
            tf.concat([coarse_points, fine_points], axis=-1), axis=-1)

        # Encode the fine rays
        logging.debug('Encoding fine rays')
        fine_rays, fine_rays_direction = encode_position_and_directions(
            ray_origin, ray_direction, fine_points,  self.pos_emb_xyz, self.pos_emb_dir)

        # Compute the fine rgb and sigma
        logging.debug('Computing fine rgb and sigma')
        fine_rgb, fine_sigma = self.predict_fine(
            fine_rays, fine_rays_direction)

        # Render the fine image and depth
        fine_rgb, fine_sigma = self.ensure_mlp_shape(
            fine_rgb, fine_sigma, self.n_coarse + self.n_fine)

        logging.debug('Rendering fine image')
        fine_image, fine_depth, fine_weights = render_image_depth(
            fine_rgb, fine_sigma, fine_points)

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


# if __name__ == '__main__':
    #     model = NeRFMLP()
    #     POS_ENCODE_DIMS = 16
    #     model.build(input_shape=[[4, 256, 256, 2 * 3 * POS_ENCODE_DIMS + 3],
    #                 [4, 256, 256, 2 * 3 * POS_ENCODE_DIMS + 3]])
    #     model.summary()
    # nerf = NeRF()
    # img_size = 100
    # ray_origin, ray_direction, sample_points = tf.random.uniform((4, img_size, img_size, 3)), tf.random.uniform(
    #     (4, img_size, img_size, 3)), tf.random.uniform((4, img_size, img_size, 64))

    # ray_coarse = (ray_origin[..., None, :] +
    #               ray_direction[..., None, :] * sample_points[..., None])

    # print('ray_coarse shape', ray_coarse.shape)

    # ray_coarse_encoded = nerf.positional_encoding(ray_coarse)
