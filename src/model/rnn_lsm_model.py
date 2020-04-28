import tensorflow as tf
from tensorflow.keras.regularizers import l2


def build_rnn_lstm_model(tokenizer, layers):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, layers * layers,input_length=1043),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layers, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))),
        tf.keras.layers.Dense(layers, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','precision','recall'])
    return model