import tensorflow as tf


class NeRFMLP(tf.keras.Model):
    def __init__(self, n_layers: int = 8, dense_units: int = 256, skip_layer=4, **kwargs):
        super(NeRFMLP, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.dense_units = dense_units
        self.skip_layer = skip_layer

        self.mlp_layers = [
            tf.keras.layers.Dense(
                units=dense_units, activation='relu', name=f"layer_{i}")
            for i in range(n_layers)
        ]

        self.sigma = tf.keras.layers.Dense(
            units=1, activation='relu', name="sigma")

        self.features = tf.keras.layers.Dense(
            units=dense_units, name='features')

        self.rgb_features = tf.keras.layers.Dense(
            units=dense_units//2, name='rgb_features')

        self.rgb = tf.keras.layers.Dense(
            units=3, activation='sigmoid', name='rgb')

    def call(self, inputs):
        ray_coordinate_inputs, direction_inputs = inputs
        nerf_mlp = ray_coordinate_inputs

        for i, layer in enumerate(self.mlp_layers):
            nerf_mlp = layer(nerf_mlp)

            if i % self.skip_layer == 0 and i > 0:
                nerf_mlp = tf.keras.layers.concatenate(
                    [nerf_mlp, ray_coordinate_inputs], axis=-1)

        sigma = self.sigma(nerf_mlp)

        features = self.features(nerf_mlp)
        features = tf.keras.layers.concatenate(
            [features, direction_inputs], axis=-1)

        rgb_features = self.rgb_features(features)

        rgb = self.rgb(rgb_features)

        return rgb, sigma
