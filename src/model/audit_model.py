import pickle

from data_loader.load_vuln_data_local import get_training_test_data_local
from model.rnn_lsm_model import build_rnn_lstm_model
from pathlib import Path
import tensorflow as tf
import os

hidden_layers = [32]
num_epochs = 15


def get_trained_model():
    model_path = str(Path(__file__).parents[2]) + os.path.sep + 'model'
    if len(os.listdir(model_path)) == 0:
        data_train_sequence, data_test_sequence, labels_train, labels_test, tokenizer = get_training_test_data_local()
        model = build_rnn_lstm_model(tokenizer, 32)
        model.fit(data_train_sequence, labels_train, epochs=num_epochs,
                  validation_data=(data_test_sequence, labels_test))
        model.save(model_path + os.path.sep + 'auditor_model', save_format='tf')
        with open(model_path + os.path.sep + 'auditor_tokenizer', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Created new model and tokenizer and saved")
    else:
        with open(model_path + os.path.sep + 'auditor_tokenizer', 'rb') as handle:
            tokenizer = pickle.load(handle)
        model = tf.keras.models.load_model(model_path + os.path.sep + 'auditor_model')
        print("Loaded saved model")
    return model, tokenizer
