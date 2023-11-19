import numpy as np
import pandas as pd
import tensorflow as tf

def positional_encoding(seq_len, d_model):
  d_model = d_model/2

  positions = np.arange(seq_len)[:, np.newaxis]
  depths = np.arange(d_model)[np.newaxis, :]/d_model

  angle_rates = 1 / (10000**depths)
  angle_rads = positions * angle_rates
  pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

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

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = d_model

        assert d_model % self.num_heads == 0, "d_model must be evenly divisible by num_heads"
        self.depth = d_model // num_heads

        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)
        self.combine_heads = tf.keras.layers.Dense(d_model)

    def split_heads(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        scaled_attention_logits = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(float(self.depth))
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        scaled_attention = tf.matmul(attention_weights, value)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (tf.shape(inputs)[0], -1, self.embed_dim))
        outputs = self.combine_heads(concat_attention)
        return outputs

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.att = MultiHeadSelfAttention(num_heads, d_model)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation="relu"), tf.keras.layers.Dense(d_model),])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class CSMOMTransformerRanker(tf.keras.Model):
    def __init__(self, num_categories, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(CSMOMTransformerRanker, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model

        self.input_layer = tf.keras.layers.Dense(self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.ffn_output = tf.keras.layers.Dense(1)

    def call(self, inputs, training):

        # # Input Embedding
        x = self.input_layer(inputs)
 
        # Encoder Stack
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        # Output FFN
        x = self.ffn_output(x)

        return x