from model.cnn_model import *
from model.rnn_gru_model import *
from model.rnn_lsm_model import *
from model.nn_model import *
from graph.plot_draw import *


def test_neural_network(hidden_layers,data_train_sequence, labels_train, data_test_sequence, labels_test, num_epochs, tokenizer):
    for layer in hidden_layers:
        print("================== RNN LSTM " + str(layer) + "=======================")
        rnn_lstm_model = build_rnn_lstm_model(tokenizer, layer)
        rnn_lstm_history = rnn_lstm_model.fit(data_train_sequence, labels_train, epochs=num_epochs, validation_data=(data_test_sequence, labels_test))
        plot_graphs(rnn_lstm_history, 'accuracy', '[Accuracy] RNN with LSTM', 'rnn_lstm_accuracy_'+str(layer)+'.png')
        plot_graphs(rnn_lstm_history, 'loss', '[Loss] RNN with LSTM', 'rnn_lstm_loss_'+str(layer)+'.png')
        print("================== END RNN LSTM " + str(layer) + "===================")
        print("================== NN FLATEN " + str(layer) + "======================")
        nn_model = build_nn_model(tokenizer, layer)
        nn_history = nn_model.fit(data_train_sequence, labels_train, epochs=num_epochs, validation_data=(data_test_sequence, labels_test))
        plot_graphs(nn_history, 'accuracy', '[Accuracy] NN flatten', 'nn_flatten_accuracy_'+str(layer)+'.png')
        plot_graphs(nn_history, 'loss', '[Loss] NN flatten', 'nn_flatten_loss_'+str(layer)+'.png')
        print("================== END NN FLATEN " + str(layer) + "==================")
        print("================== CNN  " + str(layer) + "===========================")
        cnn_model = build_cnn_model(tokenizer,layer)
        cnn_history = cnn_model.fit(data_train_sequence, labels_train, epochs=num_epochs, validation_data=(data_test_sequence, labels_test))
        plot_graphs(cnn_history, 'accuracy', '[Accuracy] CNN', 'cnn_accuracy_'+str(layer)+'.png')
        plot_graphs(cnn_history, 'loss', '[Loss] CNN', 'cnn_loss_'+str(layer)+'.png')

        print("================== END CNN  " + str(layer) + "=======================")
        print("================== RNN  GRU " + str(layer) + "=======================")
        rnn_gru_model = build_rnn_gru_model(tokenizer,layer)
        rnn_gru_history = rnn_gru_model.fit(data_train_sequence, labels_train, epochs=num_epochs, validation_data=(data_test_sequence, labels_test))
        plot_graphs(rnn_gru_history, 'accuracy', '[Accuracy] RNN with GRU', 'rnn_gru_accuracy_'+str(layer)+'.png')
        plot_graphs(rnn_gru_history, 'loss', '[Loss] RNN with GRU', 'rnn_gru_loss_'+str(layer)+'.png')
        print("================== END RNN GRU " + str(layer) + "====================")
        print("Training the model")
