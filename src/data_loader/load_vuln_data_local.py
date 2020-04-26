import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def get_training_test_data_local():
    data_train_sentence, data_test_sentence, labels_train, labels_test = load_vuln_data()
    tokenizer = prepare_tokenizer_for_training(data_train_sentence)
    data_train_sequence = prepare_sequenced_data(data_train_sentence, tokenizer)
    data_test_sequence = prepare_sequenced_data(data_test_sentence, tokenizer)
    print_info(data_train_sequence, data_test_sequence, labels_train,labels_test)
    return data_train_sequence, data_test_sequence, labels_train,labels_test, tokenizer


def print_info(data_train_sequence, data_test_sequence, labels_train,labels_test):
    print("Train set size: ", data_train_sequence.shape[0])
    unique_train_label, counts_train_label = np.unique(labels_train, return_counts=True)
    print("Train set Labels: ", str(dict(zip(unique_train_label, counts_train_label))))
    print("Test set size: ", data_test_sequence.shape[0])
    unique_test_label, counts_test_label = np.unique(labels_test, return_counts=True)
    print("Test set Labels: ", str(dict(zip(unique_test_label, counts_test_label))))

def load_vuln_data():
    data_path = str(Path(__file__).parents[2]) + os.path.sep + 'data' + os.path.sep
    vuln_files = os.listdir(data_path)
    data_set =  pd.DataFrame(columns = ['app_name', 'vuln_name', 'vuln_desc','severity'])
    grades = []
    for file in vuln_files:
        vulns = pd.read_csv(data_path + file)
        dfupdate=vulns.sample(frac=0.1)
        dfupdate.grade=1
        vulns.update(dfupdate)
        #vulns = vulns.drop(vulns.query('grade == 0').sample(frac=.4).index)
        vulns.app_name = vulns.app_name.str.replace(r'http[s]?://', '')
        vulns.app_name = vulns.app_name.str.split('/').str[0]
        vulns.app_name = vulns.app_name.str.split(':').str[0]
        data_set = data_set.append(vulns[['app_name', 'vuln_name', 'vuln_desc','severity']].copy())
        grades = np.append(grades, vulns['grade'].astype('int32'))
    data_train_sentence, data_test_sentence, labels_train, labels_test = train_test_split(data_set
                                                                                          , grades,
                                                                                          test_size=0.35,
                                                                                          random_state=None)
    labels_train = labels_train.astype(int)
    labels_test = labels_test.astype(int)
    return data_train_sentence, data_test_sentence, labels_train, labels_test


def prepare_tokenizer_for_training(train_data_sentence):
    train_data = train_data_sentence.copy()
    tokenizer = Tokenizer(filters='.!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n', num_words=50000, oov_token="<OOV>")
    data = []
    train_data['app_name'] = 'XXBOS XXAN ' + train_data['app_name'].astype(str)
    train_data['vuln_name'] = 'XXVN ' + train_data['vuln_name'].astype(str)
    train_data['vuln_desc'] = 'XXVD ' + train_data['vuln_desc'].astype(str)
    train_data['severity'] = 'XXSV ' + train_data['severity'].astype(str) + ' XXEOS'
    data = np.hstack((data, train_data['app_name'].to_numpy() ,train_data['vuln_name'].to_numpy(), train_data['vuln_desc'].to_numpy(), train_data['severity'].to_numpy()))
    tokenizer.fit_on_texts(data)
    print("Prepared dictionary for tokenizer")
    return tokenizer

def prepare_sequenced_data(setences, tokenizer):
    data = []
    setences['app_name'] = 'XXBOS XXAN ' + setences['app_name'].astype(str)
    setences['vuln_name'] = 'XXVN ' + setences['vuln_name'].astype(str)
    setences['vuln_desc'] = 'XXVD ' + setences['vuln_desc'].astype(str)
    setences['severity'] = 'XXSV ' + setences['severity'].astype(str) + ' XXEOS'
    # tokenizing app_name
    tokenized_app_name = tokenizer.texts_to_sequences(setences['app_name'].to_numpy())
    tokenized_app_name_padded = pad_sequences(tokenized_app_name, maxlen=20, padding='post')
    # Tokenizing vuln name
    tokenized_vuln_name = tokenizer.texts_to_sequences(setences['vuln_name'].to_numpy())
    tokenized_vuln_name_padded = pad_sequences(tokenized_vuln_name, maxlen=20, padding='post')
    # Tokenizing Vuln Description
    tokenized_vuln_desc = tokenizer.texts_to_sequences(setences['vuln_desc'].to_numpy())
    tokenized_vuln_desc_padded = pad_sequences(tokenized_vuln_desc, maxlen=1000, padding='post')
    # Tokenizing severity
    tokenized_vuln_severity = tokenizer.texts_to_sequences(setences['severity'].to_numpy())
    ####
    tokenized_data_set = np.concatenate((tokenized_app_name_padded,tokenized_vuln_name_padded, tokenized_vuln_desc_padded, tokenized_vuln_severity), axis=1)
    if isinstance(data, list):
        data = np.reshape(data, (0, 1043))
        data = data.astype('int32')
    data = np.vstack( (data, tokenized_data_set))
    return data

