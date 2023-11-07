import numpy as np
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

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0, "d_model must be evenly divisible by num_heads"
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights

class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        return self.dense2(self.dense1(x))

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training):
        attn_output, attn_weights = self.mha(enc_output, enc_output, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(attn_output + x)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_output + out1)

        return out2, attn_weights

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

    def call(self, x, training):

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

    def call(self, x, enc_output, training):

        for i in range(self.num_layers):
            x, block = self.dec_layers[i](x, enc_output, training)

        return x

class TSMOMEncoderDecoderTransformer(tf.keras.models.Model):
    def __init__(self, num_categories, d_model, num_layers, num_heads, dff, dropout_rate=0.1):
        super(TSMOMEncoderDecoderTransformer, self).__init__()

        self.input_embedding = ContinuousEmbedding(d_model)

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, dropout_rate)

        self.prefinal_layer = tf.keras.layers.Dense(1)
        self.final_layer = tf.keras.layers.Dense(1, activation="tanh")

    def call(self, inputs, training):

        enc_input, dec_input = inputs

        x_enc = self.input_embedding(enc_input) 
        x_dec = self.input_embedding(dec_input) 

        # Positional Encoding
        x_enc = x_enc + positional_encoding(x_enc.shape[1], x_enc.shape[2])[tf.newaxis,:x_enc.shape[1],:]
        x_dec = x_dec + positional_encoding(x_dec.shape[1], x_dec.shape[2])[tf.newaxis,:x_dec.shape[1],:]

        # Encoder Stack
        enc_output = self.encoder(x_enc, training)
        # Decoder Stack
        dec_output = self.decoder(x_dec, enc_output, training)

        # Final Output Layers
        prefinal_output = self.prefinal_layer(dec_output)
        output = self.final_layer(prefinal_output)

        return output