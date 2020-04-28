import tensorflow as tf
from tensorflow.keras.regularizers import l2

from metrics import *


def build_rnn_lstm_model(tokenizer, layers):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 32,input_length=863),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(32, activation='relu', ),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1,precision, recall])
    print("Layers: ", len(model.layers))
    return model