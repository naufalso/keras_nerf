import os
import argparse
import tensorflow as tf
import logging
from tqdm import tqdm
import imageio

from keras_nerf.model.nerf.nerf import NeRF
from keras_nerf.model.nerf.utils import positional_encoding, encode_position_and_directions, render_image_depth, fine_hierarchical_sampling
from keras_nerf.data.utils import get_focal_from_fov, pose_spherical
from keras_nerf.data.rays import RaysGenerator


def main():
    parser = argparse.ArgumentParser()
    # NeRF Dataset Directory
    parser.add_argument('--name', type=str, default='lego',
                        help='Name of the nerf model')

    # NeRF Model Parameters
    parser.add_argument('--num_coarse_samples', type=int, default=64)
    parser.add_argument('--num_fine_samples', type=int, default=128)
    parser.add_argument('--pos_emb_xyz', type=int, default=10)
    parser.add_argument('--pos_emb_dir', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_units', type=int, default=256)
    parser.add_argument('--skip_layer', type=int, default=4)
    parser.add_argument('--ray_chunks', type=int, default=1024)

    # NeRF Dataset Parameters
    parser.add_argument('--img_wh', type=int, default=64)
    parser.add_argument('--near', type=float, default=2.0)
    parser.add_argument('--far', type=float, default=6.0)
    parser.add_argument('--fov', type=float, default=0.6911112070083618)

    # NeRF Model Weights
    parser.add_argument('--model_dirs', type=str, default='model')

    # Output Directory
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--output_freq', type=int, default=1)

    args = parser.parse_args()
    logging.info(args)

    # Create camera matrix for 360 degree view
    camera_matrix = []
    for tetha in range(0, 360, args.output_freq):
        tetha = float(tetha)
        camera_matrix.append(pose_spherical(tetha, -30.0, 4.0))

    camera_matrix = tf.stack(camera_matrix, axis=0)
    logging.info(f'Camera Matrix Shape: {camera_matrix.shape}')

    # Initialize rays generator
    rays_generator = RaysGenerator(
        focal_length=get_focal_from_fov(args.fov, args.img_wh),
        image_width=args.img_wh,
        image_height=args.img_wh,
        near=args.near,
        far=args.far,
        n_sample=args.num_coarse_samples
    )

    # Convert camera matrix to rays
    tf_ds_rays = tf.data.Dataset.from_tensor_slices(camera_matrix).map(
        rays_generator
    ).batch(1)

    # Load model
    if not os.path.exists(os.path.join(args.model_dirs, f"coarse")) or \
            not os.path.exists(os.path.join(args.model_dirs, f"fine")):
        raise FileNotFoundError(
            f"Model not found for {args.model_dirs}")

    # Initialize NeRF Model
    nerf = NeRF(
        n_coarse=args.num_coarse_samples,
        n_fine=args.num_fine_samples,
        pos_emb_xyz=args.pos_emb_xyz,
        pos_emb_dir=args.pos_emb_dir,
        n_layers=args.num_layers,
        dense_units=args.num_units,
        skip_layer=args.skip_layer,
        model_path=args.model_dirs
    )

    # Compile the model
    nerf.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        batch_size=1,
        image_width=args.img_wh,
        image_height=args.img_wh,
        ray_chunks=args.ray_chunks
    )

    images = []
    depth = []
    for rays in tqdm(tf_ds_rays, total=360//args.output_freq, desc='Rendering Images'):
        # logging.info(f"Rendering image for {rays['camera_matrix']}")
        ray_origin, ray_direction, coarse_points = rays
        coarse_rays, coarse_rays_direction = encode_position_and_directions(
            ray_origin, ray_direction, coarse_points, nerf.pos_emb_xyz, nerf.pos_emb_dir)

        coarse_rgb, coarse_sigma = nerf.predict_coarse(
            coarse_rays, coarse_rays_direction)

        coarse_image, coarse_depth, coarse_weights = render_image_depth(
            coarse_rgb, coarse_sigma, coarse_points)

        # Compute the fine rays
        fine_points = fine_hierarchical_sampling(
            coarse_points, coarse_weights, nerf.n_fine)

        # Combine the coarse and fine points
        fine_points = tf.sort(
            tf.concat([coarse_points, fine_points], axis=-1), axis=-1)

        # Encode the fine rays
        fine_rays, fine_rays_direction = encode_position_and_directions(
            ray_origin, ray_direction, fine_points, nerf.pos_emb_xyz, nerf.pos_emb_dir)

        # Split the rays into batches
        fine_rgb, fine_sigma = nerf.predict_fine(
            fine_rays, fine_rays_direction)

        fine_image, fine_depth, fine_weights = render_image_depth(
            fine_rgb, fine_sigma, fine_points)

        images.append(fine_image.numpy()[0])
        depth.append(fine_depth.numpy()[0])

    # check if the output video directory exists, if it does not, then create it
    os.makedirs(args.output_dir, exist_ok=True)
    # build the video from the frames and save it to disk
    logging.info("creating the video from the frames...")

    imageio.mimwrite(os.path.join(
        args.output_dir, f"{args.name}.gif"), images, fps=20)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    main()
