import tensorflow as _tf


if hasattr(_tf, 'compat') and hasattr(_tf.compat, 'v1'):
    tf = _tf.compat.v1
    try:
        tf.disable_v2_behavior()
    except Exception:
        pass
else:
    tf = _tf


def xavier_initializer(dtype=None):
    return _tf.keras.initializers.glorot_uniform()


def xavier_initializer_conv2d(dtype=None):
    return _tf.keras.initializers.glorot_uniform()
