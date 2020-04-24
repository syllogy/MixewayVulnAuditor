import numpy as np
import pandas as pd
import src.config.properties as properties
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import json
import jsondiff


# Function loading data from localdata source by path to file
# Second argumemt is a array which will be populated with given resources, new data will be put at the end of a array
def load_data_from_local_source():
    files = [
        r'C:\Users\gsiew\IdeaProjects\MixewayVulnAuditor\data\webappvulns_grade.csv',
        r'C:\Users\gsiew\IdeaProjects\MixewayVulnAuditor\data\opensource_grade.csv',
        r'C:\Users\gsiew\IdeaProjects\MixewayVulnAuditor\data\infra_vulns_ds.csv',
        r'C:\Users\gsiew\IdeaProjects\MixewayVulnAuditor\data\code_vulns_ds.csv'
    ]
    tokenizer = create_tokenized(files)
    data = []
    grades = []
    for file in files:
        print("Loading : ", file)
        vulns = pd.read_csv(file)
        # prepare columns
        vulns['app_name'] = 'XXBOS XXAN ' + vulns['app_name'].astype(str).str.replace('[^A-Za-z0-9]+', '')
        vulns['vuln_name'] = 'XXVN ' + vulns['vuln_name'].astype(str)
        vulns['vuln_desc'] = 'XXVD ' + vulns['vuln_desc'].astype(str)
        vulns['severity'] = 'XXSV ' + vulns['severity'].astype(str) + ' XXEOS'
        # tokenizing app_name
        tokenized_app_name = tokenizer.texts_to_sequences(vulns['app_name'].to_numpy())
        # Tokenizing vuln name
        tokenized_vuln_name = tokenizer.texts_to_sequences(vulns['vuln_name'].to_numpy())
        tokenized_vuln_name_padded = pad_sequences(tokenized_vuln_name, maxlen=30)
        # Tokenizing Vuln Description
        tokenized_vuln_desc = tokenizer.texts_to_sequences(vulns['vuln_desc'].to_numpy())
        tokenized_vuln_desc_padded = pad_sequences(tokenized_vuln_desc, maxlen=500)
        # Tokenizing severity
        tokenized_vuln_severity = tokenizer.texts_to_sequences(vulns['severity'].to_numpy())
        ####
        tokenized_data_set = np.concatenate((tokenized_app_name,tokenized_vuln_name_padded, tokenized_vuln_desc_padded, tokenized_vuln_severity), axis=1)
        if isinstance(data, list):
            data = np.reshape(data, (0, 536))
            data = data.astype('int32')
        data = np.vstack( (data, tokenized_data_set))
        grades = np.append(grades, vulns['grade'].astype('int32'))
    data_train_sentence, data_test_sentence, labels_train, labels_test = train_test_split(data
                                                                                          , grades,
                                                                                          test_size=0.25,
                                                                                          random_state=42)
    labels_train = labels_train.astype(int)
    labels_test = labels_test.astype(int)
    print("Traing Data-Set size: ", data_train_sentence.shape)
    print("Traing Labels size: ", labels_train.shape)
    print("Test Data-Set size: ", data_test_sentence.shape)
    print("Test Labels size: ", labels_test.shape)

    return data_train_sentence, data_test_sentence, labels_train, labels_test, tokenizer

def create_tokenized(files):
    tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
    data = []
    for file in files:
        vulns = pd.read_csv(file)
        vulns['app_name'] = 'XXBOS XXAN ' + vulns['app_name'].astype(str).str.replace('[^A-Za-z0-9]+', '')
        vulns['vuln_name'] = 'XXVN ' + vulns['vuln_name'].astype(str)
        vulns['vuln_desc'] = 'XXVD ' + vulns['vuln_desc'].astype(str)
        vulns['severity'] = 'XXSV ' + vulns['severity'].astype(str) + ' XXEOS'
        data = np.hstack((data, vulns['app_name'].to_numpy() ,vulns['vuln_name'].to_numpy(), vulns['vuln_desc'].to_numpy(), vulns['severity'].to_numpy()))
    tokenizer.fit_on_texts(data)
    print("Prepared dictionary for tokenizer")
    return tokenizer
