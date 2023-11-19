import tensorflow as tf
import numpy as np

class ContinuousEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(ContinuousEmbedding, self).__init__()
        self.d_model = d_model

    def build(self, input_shape):
        m = input_shape[-1]
        self.dense = tf.keras.layers.Dense(self.d_model, use_bias=False, activation=None)

    def call(self, inputs):
        return self.dense(inputs)

class CategoricalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_categories, embed_dim):
        super(CategoricalEmbedding, self).__init__()
        self.num_categories = num_categories
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(self.num_categories, self.embed_dim)

    def call(self, inputs):
        return self.embedding(inputs)

def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return look_ahead_mask  # (seq_len, seq_len)

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model

    def call(self, Q, K, V, mask=None):
        # Attention score
        attn_logits = tf.matmul(Q, K, transpose_b=True)
        attn_logits = attn_logits / (tf.math.sqrt(float(self.d_model)) + 1e-9)
        
        if mask is not None:
            attn_logits += (mask * -1e9)
        
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        
        output = tf.matmul(attn_weights, V)
        return output, attn_weights

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
        self.WQ = tf.keras.layers.Dense(d_model)
        self.WK = tf.keras.layers.Dense(d_model)
        self.WV = tf.keras.layers.Dense(d_model)
        
        self.attention = ScaledDotProductAttention(d_model)
        
        self.linear = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]
        
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        output, attn_weights = self.attention(Q, K, V, mask)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.linear(concat_output)

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(d_model)])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        attn_output = self.mha(x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def binary_activation(x):
    return tf.where(x >= 0, 1., -1.)

class TSMOMDecoderTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, seq_size, dropout_rate=0.1):
        super(TSMOMDecoderTransformer, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(seq_size, d_model)

        self.decoder_layer = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        
        self.pre_output_layer = tf.keras.layers.Dense(1)

        self.output_layer = tf.keras.layers.Dense(1, activation="tanh")
        # self.output_layer = tf.keras.layers.Dense(1, activation=binary_activation)

    def call(self, inputs, training):

        # Input Embedding
        x = self.embedding(inputs)
        x = x + self.pos_encoding[:, :tf.shape(x)[1], :]

        mask = create_mask(x)
        for i in range(self.num_layers):
            x = self.decoder_layer[i](x, training, mask)

        x = self.pre_output_layer(x)
        output = self.output_layer(x)

        return output