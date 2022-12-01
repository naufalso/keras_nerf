import os
import argparse
import tensorflow as tf


from keras_nerf.model.nerf.nerf import NeRF
from keras_nerf.model.nerf.callback import NeRFTrainMonitor
from keras_nerf.data.loader import DatasetLoader

tf.random.set_seed(42)


def main():
    parser = argparse.ArgumentParser()
    # NeRF Dataset Directory
    parser.add_argument('--name', type=str, default='lego',
                        help='Name of the nerf model')
    parser.add_argument('--data_dir', type=str,
                        default='data/nerf_synthetic/lego')

    # NeRF Model Parameters
    parser.add_argument('--num_coarse_samples', type=int, default=64)
    parser.add_argument('--num_fine_samples', type=int, default=128)
    parser.add_argument('--pos_emb_xyz', type=int, default=10)
    parser.add_argument('--pos_emb_dir', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_units', type=int, default=256)
    parser.add_argument('--skip_layer', type=int, default=4)

    # NeRF Dataset Parameters
    parser.add_argument('--img_wh', type=int, default=64)
    parser.add_argument('--near', type=float, default=2.0)
    parser.add_argument('--far', type=float, default=6.0)

    # NeRF Training Parameters
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--ray_chunks', type=int, default=1024)
    parser.add_argument('--eagerly', action='store_true')

    # NeRF Logging Parameters
    parser.add_argument('--model_dirs', type=str, default='model')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_freq', type=int, default=1)

    args = parser.parse_args()

    # Configure multi-gpu training
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # Load the data
    dataset_loader = DatasetLoader(args.data_dir)

    global_batch_size = args.batch_size * strategy.num_replicas_in_sync

    train_dataset, val_dataset, test_dataset = dataset_loader.load_dataset(
        batch_size=global_batch_size,
        image_width=args.img_wh,
        image_height=args.img_wh,
        near=args.near,
        far=args.far,
        n_sample=args.num_coarse_samples
    )

    # assert args.batch_size % strategy.num_replicas_in_sync == 0, \
    #     'Batch size must be divisible by number of GPUs'

    image_area = tf.constant(args.img_wh * args.img_wh, tf.float32)

    with strategy.scope():
        # Create the model
        nerf = NeRF(
            n_coarse=args.num_coarse_samples,
            n_fine=args.num_fine_samples,
            pos_emb_xyz=args.pos_emb_xyz,
            pos_emb_dir=args.pos_emb_dir,
            n_layers=args.num_layers,
            dense_units=args.num_units,
            skip_layer=args.skip_layer,
        )

        loss_fn = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )

        def compute_distributed_loss(y_true, y_pred):
            # * (1. / global_batch_size)
            return tf.reduce_mean(loss_fn(y_true, y_pred))

        # Compile the model
        nerf.compile(
            # tf.keras.optimizers.Adam(learning_rate=args.lr),
            optimizer='adam',
            loss=compute_distributed_loss,
            batch_size=args.batch_size,
            image_width=args.img_wh,
            image_height=args.img_wh,
            ray_chunks=args.ray_chunks,
            run_eagerly=args.eagerly
        )

        # Create the callbacks
        nerf_train_monitor = NeRFTrainMonitor(
            dataset=test_dataset,
            log_dir=args.log_dir,
            batch_size=args.batch_size,
            update_freq=args.log_freq
        )

    # Train the model
    nerf.fit(
        train_dataset,
        epochs=args.num_epochs,
        validation_data=val_dataset,
        callbacks=[nerf_train_monitor]
    )

    # Save the model
    os.makedirs(args.model_dirs, exist_ok=True)
    coarse_save_path = os.path.join(args.model_dirs, f'{args.name}_coarse')
    fine_save_path = os.path.join(args.model_dirs, f'{args.name}_fine')

    nerf.coarse.save(coarse_save_path)
    nerf.fine.save(fine_save_path)


if __name__ == '__main__':
    main()
