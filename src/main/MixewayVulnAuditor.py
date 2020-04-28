from data_loader.load_vuln_data_local import *
from model.neural_network_tester import *
from model.random_forest_tester import *

data_train_sequence, data_test_sequence, labels_train,labels_test, tokenizer = get_training_test_data_local()
hidden_layers = [4, 32]
num_epochs = 10

#test_neural_network(hidden_layers,data_train_sequence, labels_train, data_test_sequence, labels_test, num_epochs, tokenizer)
test_random_forest(data_train_sequence, labels_train, data_test_sequence, labels_test)