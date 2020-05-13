import tensorflow as tf
from tensorflow.keras.regularizers import l2
from model.metrics import *


def build_nn_model(tokenizer, layers):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 32,input_length=863),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    print("Layers: ", len(model.layers))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1,precision, recall])
    return model