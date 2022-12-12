import os
import argparse
import tensorflow as tf
import logging
from tqdm import tqdm
import imageio

from keras_nerf.model.nerf.nerf import NeRF
from keras_nerf.data.utils import get_focal_from_fov, pose_spherical
from keras_nerf.data.rays import RaysGenerator


def main():
    parser = argparse.ArgumentParser()
    # NeRF Dataset Directory
    parser.add_argument('--name', type=str, default='lego',
                        help='Name of the nerf model')

    # NeRF Model Parameters
    parser.add_argument('--model_dirs', type=str, required=True)
    parser.add_argument('--ray_chunks', type=int, default=4096)

    # NeRF Dataset Parameters
    parser.add_argument('--img_wh', type=int, default=128)
    parser.add_argument('--near', type=float, default=2.0)
    parser.add_argument('--far', type=float, default=6.0)
    parser.add_argument('--fov', type=float, default=0.6911112070083618)
    parser.add_argument('--eagerly', action='store_true')
    parser.add_argument('--white_bg', action='store_true')

    # View Parameters
    parser.add_argument('--phi', type=float, default=-30.0)
    parser.add_argument('--z_translate', type=float, default=4.0)

    # Output Directory
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--output_freq', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    logging.info(args)

    # Check if the model exists
    if not os.path.exists(os.path.join(args.model_dirs, f"coarse.h5")) or \
            not os.path.exists(os.path.join(args.model_dirs, f"fine.h5")):
        raise FileNotFoundError(
            f"Model not found for {args.model_dirs}")

    # Initialize NeRF Model
    nerf = NeRF(
        model_path=args.model_dirs
    )

    # Create camera matrix for 360 degree view
    camera_matrix = []
    for tetha in range(0, 360, args.output_freq):
        tetha = float(tetha)
        camera_matrix.append(pose_spherical(tetha, args.phi, args.z_translate))

    camera_matrix = tf.stack(camera_matrix, axis=0)
    logging.info(f'Camera Matrix Shape: {camera_matrix.shape}')

    # Initialize rays generator
    rays_generator = RaysGenerator(
        focal_length=get_focal_from_fov(args.fov, args.img_wh),
        image_width=args.img_wh,
        image_height=args.img_wh,
        near=args.near,
        far=args.far,
        n_sample=nerf.n_coarse
    )

    # Convert camera matrix to rays
    tf_ds_rays = tf.data.Dataset.from_tensor_slices(camera_matrix).map(
        rays_generator
    ).batch(1)

    # Compile the model
    nerf.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        batch_size=1,
        image_width=args.img_wh,
        image_height=args.img_wh,
        ray_chunks=args.ray_chunks,
        white_background=args.white_bg,
        is_training=False
    )

    if args.eagerly:
        nerf_predictions = nerf.predict_and_render_images
    else:
        nerf_predictions = tf.function(
            nerf.predict_and_render_images, reduce_retracing=True)

    nerf.coarse.summary()
    nerf.fine.summary()

    images = []
    depth = []
    for rays in tqdm(tf_ds_rays, total=360//args.output_freq, desc='Rendering Images'):

        _, fine_results = nerf_predictions(rays)
        (fine_image, fine_depth, _) = fine_results

        images.append(fine_image.numpy()[0])
        depth.append(fine_depth.numpy()[0])

    # check if the output video directory exists, if it does not, then create it
    os.makedirs(args.output_dir, exist_ok=True)
    # build the video from the frames and save it to disk
    logging.info("creating the video from the frames...")

    imageio.mimwrite(os.path.join(
        args.output_dir, f"{args.name}.gif"), images, fps=20)


if __name__ == "__main__":
    main()
