# coding=utf-8
"""Transformer-based Siamese DirectionNet variant."""

from pano_utils import geometry
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.layers import Conv2D
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import Dropout
from tensorflow.compat.v1.keras.layers import GlobalAveragePooling1D
from tensorflow.compat.v1.keras.layers import LayerNormalization
from tensorflow.compat.v1.keras.layers import LeakyReLU
from tensorflow.compat.v1.keras.layers import UpSampling2D


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
    qkv = tf.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim])
    qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn_logits = tf.matmul(q, k, transpose_b=True) * self.scale
    attn = tf.nn.softmax(attn_logits, axis=-1)
    attn = self.dropout(attn, training=training)

    context = tf.matmul(attn, v)
    context = tf.transpose(context, [0, 2, 1, 3])
    context = tf.reshape(context, [batch_size, seq_len, self.embed_dim])
    output = self.proj(context)
    return self.dropout(output, training=training)


class TransformerBlock(keras.layers.Layer):
  """Transformer encoder block with pre-norm."""

  def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0):
    super(TransformerBlock, self).__init__()
    self.norm1 = LayerNormalization(epsilon=1e-6)
    self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
    self.norm2 = LayerNormalization(epsilon=1e-6)
    self.mlp = keras.Sequential([
        Dense(mlp_dim, activation=tf.nn.gelu),
        Dropout(dropout),
        Dense(embed_dim),
        Dropout(dropout),
    ])

  def call(self, x, training=False):
    x = x + self.attn(self.norm1(x), training=training)
    x = x + self.mlp(self.norm2(x), training=training)
    return x


class SiameseCVTEncoder(keras.Model):
  """Siamese CVT encoder that outputs a global embedding."""

  def __init__(self,
               embed_dim=256,
               num_layers=4,
               num_heads=4,
               mlp_dim=512,
               patch_size=8,
               dropout=0.1,
               regularization=0.01):
    super(SiameseCVTEncoder, self).__init__()
    self.embed_dim = embed_dim
    self.patch_size = patch_size
    self.patch_embed = Conv2D(
        embed_dim,
        patch_size,
        strides=patch_size,
        padding='same',
        kernel_regularizer=regularizers.l2(regularization))
    self.pos_drop = Dropout(dropout)
    self.blocks = [
        TransformerBlock(embed_dim, num_heads, mlp_dim, dropout=dropout)
        for _ in range(num_layers)
    ]
    self.norm = LayerNormalization(epsilon=1e-6)
    self.fusion = Dense(embed_dim, activation=tf.nn.gelu)
    self.pool = GlobalAveragePooling1D()
    self.pos_embed = None

  def build(self, input_shape):
    output_shape = self.patch_embed.compute_output_shape(input_shape)
    if output_shape[1] is None or output_shape[2] is None:
      raise ValueError('Input shape must have static spatial dimensions.')
    token_count = int(output_shape[1] * output_shape[2])
    self.pos_embed = self.add_weight(
        'pos_embed',
        shape=[1, token_count, self.embed_dim],
        initializer='zeros')
    super(SiameseCVTEncoder, self).build(input_shape)

  def _encode(self, x, training=False):
    x = self.patch_embed(x)
    batch_size = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    tokens = tf.reshape(x, [batch_size, height * width, self.embed_dim])

    tokens = tokens + self.pos_embed
    tokens = self.pos_drop(tokens, training=training)
    for block in self.blocks:
      tokens = block(tokens, training=training)
    tokens = self.norm(tokens)
    pooled = self.pool(tokens)
    return pooled

  def call(self, img1, img2, training=False):
    y1 = self._encode(img1, training=training)
    y2 = self._encode(img2, training=training)
    merged = tf.concat([y1, y2], axis=-1)
    merged = self.fusion(merged)
    return merged[:, tf.newaxis, tf.newaxis, :]


class DirectionNetCVT(keras.Model):
  """DirectionNet with transformer-based siamese encoder/decoder."""

  def __init__(self,
               n_out,
               embed_dim=256,
               num_layers=4,
               num_heads=4,
               mlp_dim=512,
               dropout=0.1,
               regularization=0.01):
    super(DirectionNetCVT, self).__init__()
    self.encoder = SiameseCVTEncoder(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        regularization=regularization)
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
              kernel_regularizer=regularizers.l2(regularization)))
      self.decoder_transformers.append(
          TransformerBlock(
              dim,
              max(1, num_heads // 2),
              max(2 * dim, mlp_dim // 2),
              dropout=dropout))
      self.decoder_norms.append(LayerNormalization(epsilon=1e-6))
      self.decoder_activations.append(LeakyReLU())
    self.down_channel = Conv2D(
        n_out, 1, kernel_regularizer=regularizers.l2(regularization))

  def _spherical_upsampling(self, x):
    return UpSampling2D(interpolation='bilinear')(
        geometry.equirectangular_padding(x, [[1, 1], [1, 1]]))

  def _transformer_2d(self, x, block, training=False):
    batch_size = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    channels = x.shape[-1]
    tokens = tf.reshape(x, [batch_size, height * width, channels])
    tokens = block(tokens, training=training)
    return tf.reshape(tokens, [batch_size, height, width, channels])

  def call(self, img1, img2, training=False):
    y = self.encoder(img1, img2, training=training)

    for conv, transformer, norm, activation in zip(
        self.decoder_convs,
        self.decoder_transformers,
        self.decoder_norms,
        self.decoder_activations):
      y = self._spherical_upsampling(y)
      y = conv(y)
      y = self._transformer_2d(y, transformer, training=training)
      y = norm(y)
      y = activation(y)
      y = y[:, 1:-1, 1:-1, :]

    return self.down_channel(y)
