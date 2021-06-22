# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Activation


class MlpBlock(Layer):
    def __init__(self, dim, hdim, activation=tf.nn.gelu, **kwargs):
        super(MlpBlock, self).__init__(**kwargs)

        self.dim = dim
        self.hdim = dim
        self.fc1 = Dense(hdim)
        self.activation = Activation(activation)
        self.fc2 = Dense(dim)

    def call(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def compute_output_shape(self, input_signature):
        return (input_signature[0], self.dim)

    def get_config(self):
        config = super(MlpBlock, self).get_config()
        config.update({"dim": self.dim, "hdim": self.hdim})
        return config


class MixerBlock(Layer):
    def __init__(self, n_patches, channel_dim, token_mixer_hidden_dim, channel_mixer_hidden_dim=None, activation=tf.nn.gelu, **kwargs):
        super(MixerBlock, self).__init__(**kwargs)

        self.n_patches = n_patches
        self.channel_dim = channel_dim
        self.token_mixer_hidden_dim = token_mixer_hidden_dim
        self.channel_mixer_hidden_dim = channel_mixer_hidden_dim
        self.activation = activation

        if not channel_mixer_hidden_dim:
            channel_mixer_hidden_dim = token_mixer_hidden_dim

        self.norm1 = LayerNormalization(axis=1)
        self.norm2 = LayerNormalization(axis=1)
        self.perm1 = Permute((2, 1))
        self.perm2 = Permute((2, 1))
        self.token_mixer = MlpBlock(
            n_patches, token_mixer_hidden_dim, name="token_mixer"
        )
        self.channel_mixer = MlpBlock(
            channel_dim, channel_mixer_hidden_dim, name="channel_mixer"
        )

    def call(self, x):
        skip_x = x
        x = self.norm1(x)
        x = self.perm1(x)
        x = self.token_mixer(x)

        x = self.perm2(x)

        x = x + skip_x
        skip_x = x

        x = self.norm2(x)
        x = self.channel_mixer(x)

        x = x + skip_x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(MixerBlock, self).get_config()
        config.update(
            {
                "n_patches": self.n_patches,
                "channel_dim": self.channel_dim,
                "token_mixer_hidden_dim": self.token_mixer_hidden_dim,
                "channel_mixer_hidden_dim": self.channel_mixer_hidden_dim,
                "activation": self.activation,
            }
        )
        return config


def MLPMixer(
    input_shape,
    num_classes,
    num_blocks,
    patch_size,
    hdim,
    tokens_mlp_dim,
    channels_mlp_dim=None,
):
    height, width, _ = input_shape

    if not channels_mlp_dim:
        channels_mlp_dim = tokens_mlp_dim

    num_patches = (height * width) // (patch_size ** 2)

    inputs = keras.Input(input_shape)
    x = inputs

    x = Conv2D(
        hdim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name="projector",
    )(x)

    x = Reshape([-1, hdim])(x)

    for _ in range(num_blocks):
        x = MixerBlock(
            n_patches=num_patches,
            channel_dim=hdim,
            token_mixer_hidden_dim=tokens_mlp_dim,
            channel_mixer_hidden_dim=channels_mlp_dim,
        )(x)

    x = GlobalAveragePooling1D()(x)

    x = LayerNormalization(name="pre_head_layer_norm")(x)
    x = Dense(num_classes, activation="softmax", name="head")(x)

    return keras.Model(inputs, x)
