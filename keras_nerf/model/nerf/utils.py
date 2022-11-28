import tensorflow as tf


@tf.function
def render_image_depth(rgb, sigma, sample_points, epsilon=1e-10):
    """
    Volumatric Rendering
    Render the image and depth from the rgb and sigma values
    """
    # Squeeze the last dimension of sigma
    sigma = sigma[..., 0]

    # Calculate the delta between sample points
    delta = sample_points[..., 1:] - sample_points[..., :-1]
    delta_shape = delta.shape[:-1] + [1]
    delta = tf.concat(
        [delta, tf.broadcast_to([epsilon], shape=delta_shape)], axis=-1
    )

    # Calculate alpha from sigma and delta
    alpha = 1.0 - tf.exp(-sigma * delta)

    # Calculate the exponential of the cumulative sum of alpha
    exp_term = 1.0 - alpha

    # Calculate transmittance and weights for each sample point
    transmittance = tf.math.cumprod(
        exp_term + epsilon, axis=-1, exclusive=True)
    weights = alpha * transmittance

    # Build the image and depth from the weights
    image = tf.reduce_sum(weights[..., None] * rgb, axis=-2)
    depth = tf.reduce_sum(weights * sample_points, axis=-1)

    return image, depth, weights


@tf.function
def positional_encoding(inputs, pos_embedding_dim):
    # Include the original coordinate [x, y, z] (3 dims)
    positions = [inputs]
    # Add sin and cos positional encoding (2 * 3 * pos_embedding_dim)
    for i in range(pos_embedding_dim):
        for fn in [tf.sin, tf.cos]:
            positions.append(fn(2.0 ** i * inputs))

    # Output Shape: [..., 2 * 3 * pos_embedding_dim + 3]
    return tf.concat(positions, axis=-1)


@tf.function
def fine_hierarchical_sampling(mid_points, weights, n_samples):
    # add a small value to the weights to prevent it from nan
    weights += 1e-5
    # normalize the weights to get the pdf
    pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
    # from pdf to cdf transformation
    cdf = tf.cumsum(pdf, axis=-1)
    # start the cdf with 0s
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)
    # get the sample points
    uShape = mid_points.shape[:-1] + [n_samples]
    u = tf.random.uniform(shape=uShape)
    # get the indices of the points of u when u is inserted into cdf in a
    # sorted manner
    indices = tf.searchsorted(cdf, u, side="right")
    # define the boundaries
    below = tf.maximum(0, indices-1)
    above = tf.minimum(cdf.shape[-1]-1, indices)
    indicesG = tf.stack([below, above], axis=-1)

    # gather the cdf according to the indices
    cdfG = tf.gather(cdf, indicesG, axis=-1,
                     batch_dims=len(indicesG.shape)-2)

    # gather the tVals according to the indices
    mid_points = tf.gather(mid_points, indicesG, axis=-1,
                           batch_dims=len(indicesG.shape)-2)
    # create the samples by inverting the cdf
    denom = cdfG[..., 1] - cdfG[..., 0]
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u - cdfG[..., 0]) / denom
    samples = (mid_points[..., 0] + t *
               (mid_points[..., 1] - mid_points[..., 0]))

    # return the samples
    return samples


@tf.function
def encode_position_and_directions(ray_origin, ray_direction, coarse_points, pos_embedding_dim):
    # Generate rays
    # Equation: ray(t) = ray_origin + t * ray_direction
    # Shape: [batch_size, image_height, image_width, n_samples, 3]
    rays_positions = (ray_origin[..., None, :] +
                      ray_direction[..., None, :] * coarse_points[..., None])

    # Positional encode the rays positions
    # Shape: [batch_size, image_height, image_width, n_samples, 3 + 2 * 3 * pos_embedding_dim]
    rays_positions = positional_encoding(rays_positions, pos_embedding_dim)

    # Positional encode the ray directions
    # Shape: [batch_size, image_height, image_width, n_samples, 3 + 2 * 3 * pos_embedding_dim]
    rays_direction_shape = tf.shape(rays_positions[..., :3])
    rays_direction = tf.broadcast_to(
        ray_direction[..., None, :], shape=rays_direction_shape)
    rays_direction = positional_encoding(rays_direction, pos_embedding_dim)

    # Return the encoded rays
    return rays_positions, rays_direction