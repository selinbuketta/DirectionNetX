# coding=utf-8
"""Transformer-based Siamese DirectionNet variant (TF1-compatible)."""

from pano_utils import geometry
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    LayerNormalization,
    LeakyReLU,
    UpSampling2D,
)

# ------------------------------------------------------------
# TF1-compatible GELU (BERT approximation)
# ------------------------------------------------------------
def gelu(x):
    return 0.5 * x * (1.0 + tf.tanh(
        tf.sqrt(2.0 / tf.constant(3.141592653589793)) *
        (x + 0.044715 * tf.pow(x, 3))
    ))


# ------------------------------------------------------------
# Multi-Head Self Attention
# ------------------------------------------------------------
class MultiHeadSelfAttention(keras.layers.Layer):
    """Multi-head self-attention for token sequences."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = Dense(embed_dim * 3)
        self.proj = Dense(embed_dim)
        self.dropout = Dropout(dropout)

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        qkv = self.qkv(x)
        qkv = tf.reshape(
            qkv,
            [batch_size, seq_len, 3, self.num_heads, self.head_dim]
        )
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn_logits, axis=-1)
        attn = self.dropout(attn, training=training)

        context = tf.matmul(attn, v)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(
            context,
            [batch_size, seq_len, self.embed_dim]
        )

        out = self.proj(context)
        return self.dropout(out, training=training)


# ------------------------------------------------------------
# Transformer Encoder Block (Pre-Norm)
# ------------------------------------------------------------
class TransformerBlock(keras.layers.Layer):
    """Transformer encoder block with pre-norm."""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, dropout=dropout
        )
        self.norm2 = LayerNormalization(epsilon=1e-6)

        self.mlp = keras.Sequential([
            Dense(mlp_dim, activation=gelu),
            Dropout(dropout),
            Dense(embed_dim),
            Dropout(dropout),
        ])

    def call(self, x, training=False):
        x = x + self.attn(self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x


# ------------------------------------------------------------
# Siamese CvT Encoder
# ------------------------------------------------------------
class SiameseCVTEncoder(keras.Model):
    """Siamese CVT encoder that outputs a global embedding."""

    def __init__(
        self,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        mlp_dim=512,
        patch_size=8,
        dropout=0.1,
        regularization=0.01,
    ):
        super(SiameseCVTEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = Conv2D(
            embed_dim,
            patch_size,
            strides=patch_size,
            padding='same',
            kernel_regularizer=regularizers.l2(regularization),
        )

        self.pos_drop = Dropout(dropout)
        self.blocks = [
            TransformerBlock(
                embed_dim, num_heads, mlp_dim, dropout=dropout
            )
            for _ in range(num_layers)
        ]
        self.norm = LayerNormalization(epsilon=1e-6)

        self.pool = GlobalAveragePooling1D()
        self.fusion = Dense(embed_dim, activation=gelu)

        self.pos_embed = None

    def _encode(self, x, training=False):
      x = self.patch_embed(x)

      b = tf.shape(x)[0]
      h = tf.shape(x)[1]
      w = tf.shape(x)[2]

      tokens = tf.reshape(x, [b, h * w, self.embed_dim])

      # Lazy positional embedding creation (SAFE)
      if self.pos_embed is None:
          self.pos_embed = self.add_weight(
              name="pos_embed",
              shape=[1, tokens.shape[1], self.embed_dim],
              initializer="zeros",
              trainable=True,
          )

      tokens = tokens + self.pos_embed
      tokens = self.pos_drop(tokens, training=training)

      for blk in self.blocks:
          tokens = blk(tokens, training=training)

      tokens = self.norm(tokens)
      pooled = self.pool(tokens)
      return pooled




    def call(self, img1, img2, training=False):
        y1 = self._encode(img1, training=training)
        y2 = self._encode(img2, training=training)

        merged = tf.concat([y1, y2], axis=-1)
        merged = self.fusion(merged)

        return merged[:, tf.newaxis, tf.newaxis, :]


"""
# ------------------------------------------------------------
# DirectionNet with Transformer Decoder
# ------------------------------------------------------------
class DirectionNetCVT(keras.Model):
    #DirectionNet with transformer-based siamese encoder/decoder.

    def __init__(
        self,
        n_out,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        mlp_dim=512,
        dropout=0.1,
        regularization=0.01,
    ):
        super(DirectionNetCVT, self).__init__()

        self.encoder = SiameseCVTEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            regularization=regularization,
        )

        self.decoder_convs = []
        self.decoder_transformers = []
        self.decoder_norms = []
        self.decoder_activations = []

        decoder_dims = [
            embed_dim,
            embed_dim // 2,
            embed_dim // 4,
            embed_dim // 8,
            embed_dim // 16,
            embed_dim // 32,
        ]

        for dim in decoder_dims:
            self.decoder_convs.append(
                Conv2D(
                    dim,
                    3,
                    padding='same',
                    kernel_regularizer=regularizers.l2(regularization),
                )
            )
            self.decoder_transformers.append(
                TransformerBlock(
                    dim,
                    max(1, num_heads // 2),
                    max(2 * dim, mlp_dim // 2),
                    dropout=dropout,
                )
            )
            self.decoder_norms.append(
                LayerNormalization(epsilon=1e-6)
            )
            self.decoder_activations.append(LeakyReLU())

        self.down_channel = Conv2D(
            n_out,
            1,
            kernel_regularizer=regularizers.l2(regularization),
        )

    def _spherical_upsampling(self, x):
        x = geometry.equirectangular_padding(x, [[1, 1], [1, 1]])
        return UpSampling2D(interpolation='bilinear')(x)

    def _transformer_2d(self, x, block, training=False):
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = x.shape[-1]

        tokens = tf.reshape(x, [b, h * w, c])
        tokens = block(tokens, training=training)
        return tf.reshape(tokens, [b, h, w, c])

    def call(self, img1, img2, training=False):
      y = self.encoder(img1, img2, training=training)

      for conv, tr, norm, act in zip(
          self.decoder_convs,
          self.decoder_transformers,
          self.decoder_norms,
          self.decoder_activations,
      ):
          y = self._spherical_upsampling(y)
          y = conv(y)
          y = self._transformer_2d(y, tr, training=training)
          y = norm(y)
          y = act(y)
          y = y[:, 1:-1, 1:-1, :]

      y = self.down_channel(y)

      # 🔧 FORCE MATCH TO GT (64x64)
      y = tf.image.resize(
          y,
          size=[64, 64],
          method=tf.image.ResizeMethod.BILINEAR,
      )

      return y
"""

# ------------------------------------------------------------
# DirectionNet with CNN-only decoder (no transformers)
# ------------------------------------------------------------
class DirectionNetCVT(keras.Model):
    """DirectionNet with transformer-based siamese encoder and CNN decoder."""

    def __init__(
        self,
        n_out,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        mlp_dim=512,
        dropout=0.1,
        regularization=0.01,
    ):
        super(DirectionNetCVT, self).__init__()

        self.encoder = SiameseCVTEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            patch_size=8,
            dropout=dropout,
            regularization=regularization,
        )

        self.decoder_convs = []
        self.decoder_activations = []

        decoder_dims = [
            embed_dim,
            embed_dim // 2,
            embed_dim // 4,
            embed_dim // 8,
            embed_dim // 16,
            embed_dim // 32,
        ]

        for dim in decoder_dims:
            self.decoder_convs.append(
                Conv2D(
                    dim,
                    3,
                    padding="same",
                    kernel_regularizer=regularizers.l2(regularization),
                )
            )
            self.decoder_activations.append(LeakyReLU())

        self.down_channel = Conv2D(
            n_out,
            1,
            kernel_regularizer=regularizers.l2(regularization),
        )

    def _spherical_upsampling(self, x):
        x = geometry.equirectangular_padding(x, [[1, 1], [1, 1]])
        return UpSampling2D(interpolation="bilinear")(x)

    def call(self, img1, img2, training=False):
        y = self.encoder(img1, img2, training=training)

        # CNN-only decoder (DirectionNet-style)
        for conv, act in zip(self.decoder_convs, self.decoder_activations):
            y = self._spherical_upsampling(y)
            y = conv(y)
            y = act(y)
            # remove the padding area introduced for equirectangular padding
            y = y[:, 1:-1, 1:-1, :]

        y = self.down_channel(y)

        # Match your GT resolution expected by losses.py (64x64)
        y = tf.image.resize(
            y,
            size=[64, 64],
            method=tf.image.ResizeMethod.BILINEAR,
        )
        return y


