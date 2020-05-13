import tensorflow as tf
from tensorflow.keras.regularizers import l2

from metrics import *


def build_rnn_gru_model(tokenizer, layers):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 64,input_length=863),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(layers, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))),
        tf.keras.layers.Dense(layers, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.Dense(layers/2, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1,precision, recall])
    return model