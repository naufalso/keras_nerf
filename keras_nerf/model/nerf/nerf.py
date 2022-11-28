import tensorflow as tf
from keras_nerf.model.nerf.mlp import NeRFMLP
from keras_nerf.model.nerf.utils import render_image_depth, positional_encoding, encode_position_and_directions, fine_hierarchical_sampling

# TODO: replace positional_encoding with encode_position_and_directions


class NeRF(tf.keras.Model):
    def __init__(self, n_coarse: int = 64, n_fine: int = 128, pos_embedding_dim: int = 10, n_layers: int = 8, dense_units: int = 256, skip_layer=4, **kwargs):
        super(NeRF, self).__init__(**kwargs)
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.pos_embedding_dim = pos_embedding_dim
        self.n_layers = n_layers
        self.dense_units = dense_units
        self.skip_layer = skip_layer

        self.coarse = NeRFMLP(
            n_layers=n_layers, dense_units=dense_units, skip_layer=skip_layer, name='coarse_nerf')
        self.fine = NeRFMLP(
            n_layers=n_layers, dense_units=dense_units, skip_layer=skip_layer, name='fine_nerf')

        self.epsilon = 1e-10

    def compile(self, optimizer, loss, **kwargs):
        super(NeRF, self).compile(**kwargs)
        self.optimizer = optimizer
        self.loss = loss

        self.coarse_optimizer = tf.keras.optimizers.get(optimizer)
        self.fine_optimizer = tf.keras.optimizers.get(optimizer)

        self.coarse_loss_tracker = tf.keras.metrics.Mean(name="coarse_loss")
        self.coarse_psnr_metric = tf.keras.metrics.Mean(name="coarse_psnr")
        self.corase_ssim_metric = tf.keras.metrics.Mean(name="coarse_ssim")

        self.fine_loss_tracker = tf.keras.metrics.Mean(name="fine_loss")
        self.fine_psnr_metric = tf.keras.metrics.Mean(name="fine_psnr")
        self.fine_ssim_metric = tf.keras.metrics.Mean(name="fine_ssim")

    def train_step(self, inputs):
        # Unpack the data.
        images, rays = inputs
        ray_origin, ray_direction, coarse_points = rays

        # Generate coarse rays
        # Equation: ray(t) = ray_origin + t * ray_direction
        coarse_rays = (
            ray_origin[..., None, :] + ray_direction[..., None, :] * coarse_points[..., None])

        # Positional encode the coarse rays
        coarse_rays = positional_encoding(coarse_rays, self.pos_embedding_dim)

        # Build the coarse direction and positional encode it
        coarse_rays_direction_shape = tf.shape(coarse_rays[..., :3])
        coarse_rays_direction = tf.broadcast_to(
            ray_direction[..., None, :], shape=coarse_rays_direction_shape)
        coarse_rays_direction = positional_encoding(
            coarse_rays_direction, self.pos_embedding_dim)

        # Keep track of the gradients for updating coarse model
        with tf.GradientTape() as coarse_tape:
            # Compute the coarse rgb and sigma
            coarse_rgb, coarse_sigma = self.coarse(
                [coarse_rays, coarse_rays_direction])

            # Render the coarse image and depth
            coarse_image, coarse_depth, coarse_weights = render_image_depth(
                coarse_rgb, coarse_sigma, coarse_points)

            # Compute the photometric loss for the coarse model
            coarse_loss = self.loss(images, coarse_image)

        # Compute middle points for fine sampling
        mid_points = 0.5 * (coarse_points[..., 1:] + coarse_points[..., :-1])

        # Apply hierarchical sampling and get the fine samples for the fine rays
        fine_points = fine_hierarchical_sampling(
            mid_points, coarse_weights, self.n_fine_samples)

        # Combine the coarse and fine points
        fine_points = tf.sort(
            tf.concat([coarse_points, fine_points], axis=-1), axis=-1)

        # Build the fine rays
        fine_rays = (
            ray_origin[..., None, :] + ray_direction[..., None, :] * fine_points[..., None])

        # Positional encode the fine rays
        fine_rays = positional_encoding(fine_rays, self.pos_embedding_dim)

        # Buld the fine direction and positional encode it
        fine_rays_direction_shape = tf.shape(fine_rays[..., :3])
        fine_rays_direction = tf.broadcast_to(
            ray_direction[..., None, :], shape=fine_rays_direction_shape)
        fine_rays_direction = positional_encoding(
            fine_rays_direction, self.pos_embedding_dim)

        # Keep track of the gradients for updating fine model
        with tf.GradientTape() as fine_tape:
            # Compute the fine rgb and sigma
            fine_rgb, fine_sigma = self.fine(
                [fine_rays, fine_rays_direction])

            # Render the fine image and depth
            fine_image, fine_depth, fine_weights = render_image_depth(
                fine_rgb, fine_sigma, fine_points)

            # Compute the photometric loss for fine model
            fine_loss = self.loss(images, fine_image)

        # Update the model weights using backpropagation
        coarse_gradients = coarse_tape.gradient(
            coarse_loss, self.coarse.trainable_variables)
        self.coarse_optimizer.apply_gradients(
            zip(coarse_gradients, self.coarse.trainable_variables))

        fine_gradients = fine_tape.gradient(
            fine_loss, self.fine.trainable_variables)
        self.fine_optimizer.apply_gradients(
            zip(fine_gradients, self.fine.trainable_variables))

        # Compute the PSNR and SSIM metrics
        coarse_psnr = tf.image.psnr(images, coarse_image, max_val=1.0)
        coarse_ssim = tf.image.ssim(images, coarse_image, max_val=1.0)
        fine_psnr = tf.image.psnr(images, fine_image, max_val=1.0)
        fine_ssim = tf.image.ssim(images, fine_image, max_val=1.0)

        # Update the loss and metrics trackers
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
