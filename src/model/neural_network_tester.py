from model.cnn_model import *
from model.rnn_gru_model import *
from model.rnn_lsm_model import *
from model.nn_model import *
from graph.plot_draw import *


def test_neural_network(hidden_layers,data_train_sequence, labels_train, data_test_sequence, labels_test, num_epochs, tokenizer):
    metrics = ['accuracy','precision','recall','loss']
    for layer in hidden_layers:
        print("================== RNN LSTM " + str(layer) + "=======================")
        rnn_lstm_model = build_rnn_lstm_model(tokenizer, layer)
        rnn_lstm_history = rnn_lstm_model.fit(data_train_sequence, labels_train, epochs=num_epochs, validation_data=(data_test_sequence, labels_test))
        for metric in metrics:
            plot_graphs(rnn_lstm_history, metric, '['+metric+'] RNN with LSTM', 'rnn_lstm_'+metric+'_'+str(layer)+'.png')
        print("================== END RNN LSTM " + str(layer) + "===================")
        print("================== NN FLATEN " + str(layer) + "======================")
        nn_model = build_nn_model(tokenizer, layer)
        nn_history = nn_model.fit(data_train_sequence, labels_train, epochs=num_epochs, validation_data=(data_test_sequence, labels_test))
        for metric in metrics:
            plot_graphs(nn_history, metric, '['+metric+'] NN', 'nn_'+metric+'_'+str(layer)+'.png')
        print("================== END NN FLATEN " + str(layer) + "==================")
        print("================== CNN  " + str(layer) + "===========================")
        cnn_model = build_cnn_model(tokenizer,layer)
        cnn_history = cnn_model.fit(data_train_sequence, labels_train, epochs=num_epochs, validation_data=(data_test_sequence, labels_test))
        for metric in metrics:
            plot_graphs(cnn_history, metric, '['+metric+'] CNN', 'cnn_'+metric+'_'+str(layer)+'.png')
        print("================== END CNN  " + str(layer) + "=======================")
        print("================== RNN  GRU " + str(layer) + "=======================")
        rnn_gru_model = build_rnn_gru_model(tokenizer,layer)
        rnn_gru_history = rnn_gru_model.fit(data_train_sequence, labels_train, epochs=num_epochs, validation_data=(data_test_sequence, labels_test))
        for metric in metrics:
            plot_graphs(rnn_gru_history, metric, '['+metric+'] RNN with GRU', 'rnn_gru_'+metric+'_'+str(layer)+'.png')
        print("================== END RNN GRU " + str(layer) + "====================")
        plot_combined_recall(rnn_lstm_history, rnn_gru_history, nn_history,cnn_history, "neural_network_recall.png");
        plot_combined_precision(rnn_lstm_history, rnn_gru_history, nn_history,cnn_history, "neural_network_precision.png");
