import os
import logging
import tensorflow as tf
import matplotlib.pyplot as plt

from keras_nerf.model.nerf.utils import render_image_depth, encode_position_and_directions, fine_hierarchical_sampling


class NeRFTrainMonitor(tf.keras.callbacks.Callback):
    def __init__(self, dataset: tf.data.Dataset, log_dir: str, batch_size: int, update_freq: int = 1, **kwargs):
        super(NeRFTrainMonitor, self).__init__(**kwargs)
        logging.info('Initializing NeRFTrainMonitor')
        logging.info(
            f'Log Directory: {log_dir}, Batch Size: {batch_size}, Update Frequency: {update_freq}')
        self.dataset = dataset
        self.log_dir = log_dir
        self.batch_size = batch_size

        self.update_freq = update_freq
        self.log_model_dir = os.path.join(log_dir, 'model')
        os.makedirs(self.log_model_dir, exist_ok=True)

        # self.log_sample = log_sample

        self.coarse_log_list = []
        self.val_coarse_log_list = []
        self.fine_log_list = []
        self.val_fine_log_list = []

        os.makedirs(self.log_dir, exist_ok=True)

        for inputs in self.dataset.take(1):
            self.images, self.rays = inputs
            self.ray_origin, self.ray_direction, self.coarse_points = self.rays
            self.ray_origin, self.ray_direction, self.coarse_points = self.ray_origin[
                :batch_size], self.ray_direction[:batch_size], self.coarse_points[:batch_size]
            print(self.ray_origin.shape, self.ray_direction.shape,
                  self.coarse_points.shape)

    def on_epoch_end(self, epoch, logs=None):
        self.coarse_log_list.append(logs['coarse_loss'])
        self.val_coarse_log_list.append(logs['val_coarse_loss'])
        self.fine_log_list.append(logs['fine_loss'])
        self.val_fine_log_list.append(logs['val_fine_loss'])

        if epoch % self.update_freq == 0:
            self.coarse_rays, self.coarse_rays_direction = encode_position_and_directions(
                self.ray_origin, self.ray_direction, self.coarse_points, self.model.pos_emb_xyz, self.model.pos_emb_dir)

            coarse_rgb, coarse_sigma = self.model.predict_coarse(
                self.coarse_rays, self.coarse_rays_direction)

            coarse_image, coarse_depth, coarse_weights = render_image_depth(
                coarse_rgb, coarse_sigma, self.coarse_points)

            # Compute the fine rays
            fine_points = fine_hierarchical_sampling(
                self.coarse_points, coarse_weights, self.model.n_fine)

            # Combine the coarse and fine points
            fine_points = tf.sort(
                tf.concat([self.coarse_points, fine_points], axis=-1), axis=-1)

            # Encode the fine rays
            fine_rays, fine_rays_direction = encode_position_and_directions(
                self.ray_origin, self.ray_direction, fine_points, self.model.pos_emb_xyz, self.model.pos_emb_dir)

            # Split the rays into batches
            fine_rgb, fine_sigma = self.model.predict_fine(
                fine_rays, fine_rays_direction)

            fine_image, fine_depth, fine_weights = render_image_depth(
                fine_rgb, fine_sigma, fine_points)

            # Plot the test images
            for i in range(self.batch_size):
                fig = plt.figure(figsize=(20, 10))
                gs = fig.add_gridspec(2, 5)
                # fig, axs = plt.subplots(3, 2, figsize=(16, 16))

                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(coarse_image[i])
                ax1.set_title('Coarse Image')

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(coarse_depth[i])
                ax2.set_title('Coarse Depth')

                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(fine_image[i])
                ax3.set_title('Fine Image')

                ax4 = fig.add_subplot(gs[0, 3])
                ax4.imshow(fine_depth[i])
                ax4.set_title('Fine Depth')

                ax4 = fig.add_subplot(gs[0, 4])
                ax4.imshow(self.images[i, ..., :3])
                ax4.set_title('Ground Truth')

                ax5 = fig.add_subplot(gs[1, :])
                ax5.plot(self.coarse_log_list, color='blue',
                         label='Coarse Train Loss')
                ax5.plot(self.val_coarse_log_list, color='blue',
                         linestyle='dashed', label='Coarse Val Loss')
                ax5.plot(self.fine_log_list, color='orange',
                         label='Fine Train Loss')
                ax5.plot(self.val_fine_log_list, color='orange',
                         linestyle='dashed', label='Fine Val Loss')
                ax5.legend()
                ax5.set_title(f'Loss Plot: {epoch}')

                plt.savefig(os.path.join(
                    self.log_dir, f'test_{i}_{epoch}.png'))
                plt.close()

            # Plot last train images
            for i in range(self.batch_size):
                fig = plt.figure(figsize=(20, 5))
                gs = fig.add_gridspec(1, 5)

                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(self.model.last_train_coarse_image[i])
                ax1.set_title('Coarse Image')

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(self.model.last_train_coarse_depth[i])
                ax2.set_title('Coarse Depth')

                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(self.model.last_train_fine_image[i])
                ax3.set_title('Fine Image')

                ax4 = fig.add_subplot(gs[0, 3])
                ax4.imshow(self.model.last_train_fine_depth[i])
                ax4.set_title('Fine Depth')

                ax5 = fig.add_subplot(gs[0, 4])
                ax5.imshow(self.model.last_train_image[i])
                ax5.set_title('Ground Truth')

                plt.savefig(os.path.join(
                    self.log_dir, f'train_{i}_{epoch}.png'))
                plt.close()

            # Save the model
            self.model.coarse.save(os.path.join(self.log_model_dir, f'coarse'))
            self.model.fine.save(os.path.join(self.log_model_dir, f'fine'))
