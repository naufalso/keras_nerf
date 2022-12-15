import os
import argparse
import tensorflow as tf
import logging

from keras_nerf.model.nerf.nerf import NeRF
from keras_nerf.model.nerf.callback import NeRFTrainMonitor
from keras_nerf.data.loader import DatasetLoader

tf.random.set_seed(42)


def main():
    # Tune --ray_chunks to fit your GPU memory

    # Tested on DGX Station Tesla V100 32GB
    # --img_wh 128 --ray_chunks 2048 -> Verified (3s per step)
    # --eagerly --img_wh 128 --ray_chunks 4096 -> Verified (2s per step)
    # --eagerly --img_wh 128 --ray_chunks 4096 --batch_size 4 -> Verified (11s per step)
    # --eagerly --img_wh 512 --ray_chunks 4096: larger ray_chunks will cause OOM

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
    parser.add_argument('--img_wh', type=int, default=128)
    parser.add_argument('--near', type=float, default=2.0)
    parser.add_argument('--far', type=float, default=6.0)
    parser.add_argument('--white_bg', action='store_true')

    # NeRF Training Parameters
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ray_chunks', type=int, default=2048)
    parser.add_argument('--eagerly', action='store_true')

    # NeRF Logging Parameters
    parser.add_argument('--model_dirs', type=str, default='model')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s (%(filename)s:%(lineno)d)"
    )

    logging.info(args)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Load the data
    dataset_loader = DatasetLoader(args.data_dir, args.white_bg)
    train_dataset, val_dataset, test_dataset = dataset_loader.load_dataset(
        batch_size=args.batch_size,
        image_width=args.img_wh,
        image_height=args.img_wh,
        near=args.near,
        far=args.far,
        n_sample=args.num_coarse_samples
    )

    # Create the model
    last_model_path = os.path.join(args.log_dir, args.name, "model")
    if os.path.exists(os.path.join(last_model_path, "coarse.h5")) and \
            os.path.exists(os.path.join(last_model_path, "fine.h5")):
        logging.info("Loading the latest logged model")
        model_path = last_model_path
    else:
        model_path = None

    tf.keras.backend.clear_session()
    tf.config.run_functions_eagerly(args.eagerly and args.verbose)

    nerf = NeRF(
        n_coarse=args.num_coarse_samples,
        n_fine=args.num_fine_samples,
        pos_emb_xyz=args.pos_emb_xyz,
        pos_emb_dir=args.pos_emb_dir,
        n_layers=args.num_layers,
        dense_units=args.num_units,
        skip_layer=args.skip_layer,
        model_path=model_path
    )

    # Create the callbacks
    nerf_train_monitor = NeRFTrainMonitor(
        dataset=test_dataset,
        log_dir=os.path.join(args.log_dir, args.name),
        batch_size=args.batch_size,
        update_freq=args.log_freq,
        verbose=args.verbose
    )

    last_epoch = nerf_train_monitor.last_epoch
    logging.info("Last epoch: {}".format(last_epoch))

    # Compile the model
    nerf.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        batch_size=args.batch_size,
        image_width=args.img_wh,
        image_height=args.img_wh,
        ray_chunks=args.ray_chunks,
        run_eagerly=args.eagerly,
        white_background=args.white_bg
    )

    # Train the model
    nerf.fit(
        train_dataset,
        epochs=args.num_epochs,
        validation_data=val_dataset,
        callbacks=[nerf_train_monitor],
        initial_epoch=last_epoch
    )

    # Save the model
    os.makedirs(args.model_dirs, exist_ok=True)
    save_path = os.path.join(args.model_dirs, args.name)
    nerf.save_model(save_path)


if __name__ == '__main__':
    main()
