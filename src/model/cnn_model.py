import tensorflow as tf
from metrics import *
from tensorflow.keras.regularizers import l2

def build_cnn_model(tokenizer, layers):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 32, input_length=863),
        tf.keras.layers.Conv1D(32, 5, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1,precision, recall])
    return model