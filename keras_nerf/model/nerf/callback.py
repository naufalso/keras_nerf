import os
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from csv import DictWriter, DictReader


class NeRFTrainMonitor(tf.keras.callbacks.Callback):
    def __init__(self, dataset: tf.data.Dataset, log_dir: str, batch_size: int, update_freq: int = 1, verbose: bool = False, ** kwargs):
        super(NeRFTrainMonitor, self).__init__(**kwargs)
        logging.info('Initializing NeRFTrainMonitor')
        logging.info(
            f'Log Directory: {log_dir}, Batch Size: {batch_size}, Update Frequency: {update_freq}')
        self.dataset = dataset
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.verbose = verbose

        self.log_model_dir = os.path.join(log_dir, 'model')
        os.makedirs(self.log_model_dir, exist_ok=True)

        self.coarse_log_list = []
        self.val_coarse_log_list = []
        self.fine_log_list = []
        self.val_fine_log_list = []

        if self.verbose:
            self.coarse_log_list_batch = []
            self.fine_log_list_batch = []

        # Read the last log file
        self.last_epoch = 0
        self.log_csv = os.path.join(log_dir, 'log.csv')
        if os.path.exists(self.log_csv):
            with open(self.log_csv, 'r') as f:
                csv_reader = DictReader(f)
                for i, row in enumerate(csv_reader):
                    if i > 0:
                        self.coarse_log_list.append(float(row['coarse_loss']))
                        self.val_coarse_log_list.append(
                            float(row['val_coarse_loss']))
                        self.fine_log_list.append(float(row['fine_loss']))
                        self.val_fine_log_list.append(
                            float(row['val_fine_loss']))
                        self.last_epoch = int(row['epoch'])
            self.last_epoch += 1

        # self.log_sample = log_sample

        os.makedirs(self.log_dir, exist_ok=True)

        for inputs in self.dataset.take(1):
            self.images, self.rays = inputs
            ray_origin, ray_direction, coarse_points = self.rays
            (self.ray_origin, self.ray_direction, self.coarse_points) = (
                ray_origin[:self.batch_size], ray_direction[:self.batch_size], coarse_points[:self.batch_size])

        self.dataset_iterator = iter(self.dataset)
        self.dataset_iterator.get_next()

    def on_train_batch_end(self, batch, logs=None):
        if self.verbose:
            logging.debug(f'Batch {batch}: {logs}')
            self.coarse_log_list_batch.append(logs['coarse_loss'])
            self.fine_log_list_batch.append(logs['fine_loss'])

            coarse_results, fine_results = self.model.predict_and_render_images(
                (self.ray_origin, self.ray_direction, self.coarse_points))

            coarse_image, coarse_depth = coarse_results['image'], coarse_results['depth']
            fine_image, fine_depth = fine_results['image'], fine_results['depth']

            # Plot the test images
            for i in range(self.batch_size):
                fig = plt.figure(figsize=(20, 10))
                gs = fig.add_gridspec(2, 5)
                # fig, axs = plt.subplots(3, 2, figsize=(16, 16))

                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(coarse_image[i])
                ax1.set_title('Coarse Image')

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(coarse_depth[i], cmap='inferno')
                ax2.set_title('Coarse Depth')

                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(fine_image[i])
                ax3.set_title('Fine Image')

                ax4 = fig.add_subplot(gs[0, 3])
                ax4.imshow(fine_depth[i], cmap='inferno')
                ax4.set_title('Fine Depth')

                ax4 = fig.add_subplot(gs[0, 4])
                ax4.imshow(self.images[i, ..., :3])
                ax4.set_title('Ground Truth')

                ax5 = fig.add_subplot(gs[1, :])
                ax5.plot(self.coarse_log_list_batch, color='blue',
                         label='Coarse Train Loss')
                ax5.plot(self.fine_log_list_batch, color='orange',
                         label='Fine Train Loss')
                ax5.legend()
                ax5.set_yscale('log')
                ax5.set_title(f'Loss Batch Plot: {batch}')

                plt.savefig(os.path.join(
                    self.log_dir, f'debug_{i}_{batch}.png'))
                plt.close()

    def on_epoch_end(self, epoch, logs):
        self.coarse_log_list.append(logs['coarse_loss'])
        self.val_coarse_log_list.append(logs['val_coarse_loss'])
        self.fine_log_list.append(logs['fine_loss'])
        self.val_fine_log_list.append(logs['val_fine_loss'])

        if epoch % self.update_freq == 0:
            coarse_results, fine_results = self.model.predict_and_render_images(
                (self.ray_origin, self.ray_direction, self.coarse_points))

            coarse_image, coarse_depth = coarse_results['image'], coarse_results['depth']
            fine_image, fine_depth = fine_results['image'], fine_results['depth']

            # Plot the test images
            for i in range(self.batch_size):
                fig = plt.figure(figsize=(20, 10))
                gs = fig.add_gridspec(2, 5)

                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(coarse_image[i])
                ax1.set_title('Coarse Image')

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(coarse_depth[i], cmap='inferno')
                ax2.set_title('Coarse Depth')

                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(fine_image[i])
                ax3.set_title('Fine Image')

                ax4 = fig.add_subplot(gs[0, 3])
                ax4.imshow(fine_depth[i], cmap='inferno')
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
                ax5.set_yscale('log')
                ax5.set_title(f'Loss Plot: {epoch}')

                plt.savefig(os.path.join(
                    self.log_dir, f'test_{i}_{epoch}.png'))
                plt.close()

            # Predict other test images
            inputs = self.dataset_iterator.get_next()
            images, rays = inputs
            images = images[..., :3]

            ray_origin, ray_direction, coarse_points = rays
            (ray_origin, ray_direction, coarse_points) = (
                ray_origin[:self.batch_size], ray_direction[:self.batch_size], coarse_points[:self.batch_size])

            coarse_results, fine_results = self.model.predict_and_render_images(
                (ray_origin, ray_direction, coarse_points))

            coarse_image, coarse_depth = coarse_results['image'], coarse_results['depth']
            fine_image, fine_depth = fine_results['image'], fine_results['depth']

            for i in range(self.batch_size):
                fig = plt.figure(figsize=(20, 5))
                gs = fig.add_gridspec(1, 5)

                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(coarse_image[i])
                ax1.set_title('Coarse Image')

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(coarse_depth[i], cmap='inferno')
                ax2.set_title('Coarse Depth')

                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(fine_image[i])
                ax3.set_title('Fine Image')

                ax4 = fig.add_subplot(gs[0, 3])
                ax4.imshow(fine_depth[i], cmap='inferno')
                ax4.set_title('Fine Depth')

                ax5 = fig.add_subplot(gs[0, 4])
                ax5.imshow(images[i])
                ax5.set_title('Ground Truth')

                plt.savefig(os.path.join(
                    self.log_dir, f'test_sample_{i}_{epoch}.png'))
                plt.close()

            # Write training logs into csv file
            with open(self.log_csv, 'a') as f:
                new_logs = {'epoch': epoch}
                new_logs.update(logs)
                dict_writer = DictWriter(f, new_logs.keys())
                if epoch == 0:
                    dict_writer.writeheader()
                dict_writer.writerow(new_logs)

            # Save the model
            self.model.save_model(self.log_model_dir,
                                  weights_only=(epoch != 0))

        if self.verbose:
            self.coarse_log_list_batch = []
            self.fine_log_list_batch = []
