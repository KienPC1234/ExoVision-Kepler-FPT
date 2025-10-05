import tensorflow as tf

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU available:")
    for gpu in gpus:
        print("  -", gpu)
else:
    print("❌ No GPU found, running on CPU")

import time
import numpy as np

with tf.device('/GPU:0'):
    a = tf.random.normal([10000, 1000])
    b = tf.random.normal([1000, 2000])

    start = time.time()
    c = tf.matmul(a, b)
    print("Matrix multiply result shape:", c.shape)
    print("Execution time on GPU:", time.time() - start, "seconds")
