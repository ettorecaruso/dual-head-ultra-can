import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

_HE = keras.initializers.HeNormal()

N_SEQ = 100
N_FEATURES = 3


class ReceivedSlice(layers.Layer):
    def call(self, inputs):
        return inputs[..., :2]


class AttentionPool1D(layers.Layer):
    def call(self, inputs):
        features, weights = inputs
        return tf.reduce_sum(features * weights, axis=1)


class PositionFeature(layers.Layer):
    def call(self, features):
        magnitude = tf.abs(features)
        denom = tf.reduce_sum(magnitude, axis=1, keepdims=True) + 1e-8
        positions = tf.range(tf.shape(features)[1], dtype=tf.float32)[None, :, None] / tf.cast(
            tf.shape(features)[1], tf.float32)
        return tf.reduce_sum(positions * magnitude, axis=1) / tf.squeeze(denom, axis=1)


class DelayProfile(layers.Layer):
    def __init__(self, max_lag=33, **kwargs):
        super().__init__(**kwargs)
        self.max_lag = max_lag

    def call(self, inputs):
        received = tf.cast(inputs[..., 0], tf.float32)
        received_i = tf.cast(inputs[..., 1], tf.float32)
        reference = tf.cast(inputs[..., 2], tf.float32)
        length = tf.shape(reference)[1]
        cols = []
        for lag in range(1, self.max_lag + 1):
            cols.append(self._normalized_corr(received, reference, lag))
            cols.append(self._normalized_corr(received_i, reference, lag))
            cols.append(self._normalized_corr(reference, reference, lag))
        return tf.stack(cols, axis=1)

    def _normalized_corr(self, signal, reference, lag):
        length = tf.shape(signal)[1]
        base = signal[:, lag:]
        shifted = reference[:, : length - lag]
        num = tf.reduce_sum(base * shifted, axis=1)
        den = tf.sqrt(
            tf.reduce_sum(base * base, axis=1) * tf.reduce_sum(shifted * shifted, axis=1)
        ) + 1e-8
        return num / den

    def get_config(self):
        config = super().get_config()
        config.update({"max_lag": self.max_lag})
        return config


class MultiHeadQKVAttention(layers.Layer):
    def __init__(self, attention_dim=64, num_heads=8, positional_encoding=True, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.positional_encoding = positional_encoding
        self.head_dim = attention_dim // num_heads
        self.dense_q = layers.Dense(attention_dim, kernel_initializer=_HE, name="q_dense")
        self.dense_k = layers.Dense(attention_dim, kernel_initializer=_HE, name="k_dense")
        self.dense_v = layers.Dense(attention_dim, kernel_initializer=_HE, name="v_dense")
        self.dense_o = layers.Dense(attention_dim, kernel_initializer=_HE, name="o_dense")
        self.attention_weights = None

    def call(self, inputs):
        sequence_length = tf.shape(inputs)[1]
        if self.positional_encoding:
            encoding = self._sinusoidal(sequence_length)
            inputs = inputs + tf.cast(encoding, inputs.dtype)
        q = self.dense_q(inputs)
        k = self.dense_k(inputs)
        v = self.dense_v(inputs)
        batch = tf.shape(q)[0]
        reshape = (batch, sequence_length, self.num_heads, self.head_dim)
        q = tf.reshape(q, reshape)
        k = tf.reshape(k, reshape)
        v = tf.reshape(v, reshape)
        scores = tf.einsum("bthd,bThd->bhtT", q, k) / tf.sqrt(
            tf.cast(self.head_dim, q.dtype)
        )
        weights = tf.nn.softmax(scores, axis=-1)
        self.attention_weights = tf.reduce_mean(weights, axis=1)
        context = tf.einsum("bhtT,bThd->bthd", weights, v)
        context = tf.reshape(context, (batch, sequence_length, self.attention_dim))
        output = self.dense_o(context) + inputs
        return output

    def _sinusoidal(self, length):
        positions = tf.range(length, dtype=tf.float32)[:, None]
        freqs = tf.pow(
            10000.0, -tf.range(0, self.attention_dim, 2, dtype=tf.float32) / self.attention_dim
        )
        angles = positions * freqs
        enc = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
        if self.attention_dim % 2:
            enc = tf.concat([enc, tf.zeros((length, 1))], axis=-1)
        return enc[None, ...]

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.attention_dim,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "attention_dim": self.attention_dim,
                "num_heads": self.num_heads,
                "positional_encoding": self.positional_encoding,
            }
        )
        return config


def _communication_head():
    inp = keras.Input(shape=(64,), name="comm_features")
    x = layers.Dropout(0.3, name="dropout_comm")(inp)
    x = layers.Dense(128, activation="relu", kernel_initializer=_HE, name="dense_comm_1")(x)
    logits = layers.Dense(2, name="comm_logits")(x)
    return keras.Model(inp, logits, name="communication_head")


def _sensing_head(dimension):
    inp = keras.Input(shape=(dimension,), name="sensing_features")
    x = layers.Dense(32, activation="relu", kernel_initializer=_HE, name="dense_sensing_1")(inp)
    output = layers.Dense(2, name="sensing_out")(x)
    return keras.Model(inp, output, name="sensing_head")


def _conv_stack(received):
    x = layers.Conv1D(32, 3, padding="same", activation="relu",
                      kernel_initializer=_HE, name="conv1")(received)
    x = layers.Conv1D(64, 3, padding="same", activation="relu",
                      kernel_initializer=_HE, name="conv2")(x)
    return x


def _attach_heads(pooled, features):
    comm_model = _communication_head()
    sensing_model = _sensing_head(int(features.shape[-1]))
    comm = layers.Activation("linear", name="comm_logits")(comm_model(pooled))
    sensing = layers.Activation("linear", name="sensing_out")(sensing_model(features))
    return comm, sensing


def build_conv1d_ultra_can():
    inputs = keras.Input(shape=(N_SEQ, N_FEATURES), name="isac_input")
    received = ReceivedSlice(name="received_slice")(inputs)
    hidden = _conv_stack(received)
    projected = layers.Conv1D(64, 1, kernel_initializer=_HE, name="pool_projection")(hidden)
    scores = layers.Dense(1, kernel_initializer=_HE, name="attn_scores")(projected)
    weights = layers.Softmax(axis=1, name="attn_weights")(scores)
    pooled = AttentionPool1D(name="attn_pool")([hidden, weights])
    position = PositionFeature(name="sensing_position")(hidden)
    profile = DelayProfile(name="sensing_delay_profile")(inputs)
    features = layers.Concatenate(name="sensing_features")([pooled, position, profile])
    comm, sensing = _attach_heads(pooled, features)
    return keras.Model(inputs, [comm, sensing], name="dual_head_ultra_can")


def build_qkv_ultra_can():
    inputs = keras.Input(shape=(N_SEQ, N_FEATURES), name="isac_input")
    received = ReceivedSlice(name="received_slice")(inputs)
    hidden = _conv_stack(received)
    attention = MultiHeadQKVAttention(name="qkv_attention")(hidden)
    pooled = layers.GlobalAveragePooling1D(name="shared_gap")(attention)
    pooled = layers.Dense(64, kernel_initializer=_HE, name="pool_projection")(pooled)
    position = PositionFeature(name="sensing_position")(hidden)
    profile = DelayProfile(name="sensing_delay_profile")(inputs)
    features = layers.Concatenate(name="sensing_features")([pooled, position, profile])
    comm, sensing = _attach_heads(pooled, features)
    return keras.Model(inputs, [comm, sensing], name="dual_head_ultra_can_qkv")


def build_lstm_baseline():
    inputs = keras.Input(shape=(N_SEQ, N_FEATURES), name="isac_input")
    received = ReceivedSlice(name="received_slice")(inputs)
    x = layers.LSTM(77, return_sequences=True, name="lstm_1")(received)
    x = layers.Dropout(rate=0.5, name="dropout_lstm")(x)
    pooled = layers.GlobalAveragePooling1D(name="shared_gap")(x)
    representation = layers.Dense(64, kernel_initializer=_HE, name="lstm_proj")(pooled)
    position = PositionFeature(name="sensing_position")(x)
    profile = DelayProfile(name="sensing_delay_profile")(inputs)
    features = layers.Concatenate(name="sensing_features")([position, profile, representation])
    comm, sensing = _attach_heads(representation, features)
    return keras.Model(inputs, [comm, sensing], name="lstm_baseline")


def build_mc_dlsk_baseline():
    inputs = keras.Input(shape=(N_SEQ, N_FEATURES), name="isac_input")
    received = ReceivedSlice(name="received_slice")(inputs)
    x = layers.Bidirectional(
        layers.LSTM(50, return_sequences=True), name="mc_bilstm_1"
    )(received)
    x = layers.Dropout(rate=0.5, name="dropout_mc")(x)
    pooled = layers.GlobalAveragePooling1D(name="shared_gap")(x)
    representation = layers.Dense(64, kernel_initializer=_HE, name="mc_proj")(pooled)
    position = PositionFeature(name="sensing_position")(x)
    profile = DelayProfile(name="sensing_delay_profile")(inputs)
    features = layers.Concatenate(name="sensing_features")([position, profile, representation])
    comm, sensing = _attach_heads(representation, features)
    return keras.Model(inputs, [comm, sensing], name="mc_dlsk_baseline")


def build_model(model_type: str):
    builders = {
        "conv1d": build_conv1d_ultra_can,
        "qkv": build_qkv_ultra_can,
        "lstm": build_lstm_baseline,
        "mc_dlsk": build_mc_dlsk_baseline,
    }
    return builders[model_type]()
