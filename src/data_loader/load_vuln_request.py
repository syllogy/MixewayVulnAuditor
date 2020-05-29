from pandas import np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_vuln_from_request(request, tokenizer):
    APP_NAME = 'XXBOS XXAN ' + request.appName
    APP_CONTEXT = 'XXAC ' + request.appContext
    VULN_NAME = 'XXVN ' + request.vulnName
    VULN_DESC = 'XXVD ' + request.vulnDescription
    SEVERITY = 'XXSV ' + request.severity + ' XXEOS'
    # tokenizing app_name
    tokenized_app_name = tokenizer.texts_to_sequences([APP_NAME])
    tokenized_app_name_padded = pad_sequences(tokenized_app_name, maxlen=20, padding='post')
    # tokenizing app_context
    tokenized_app_context = tokenizer.texts_to_sequences([APP_CONTEXT])
    tokenized_app_context_padded = pad_sequences(tokenized_app_name, maxlen=20, padding='post')
    # Tokenizing vuln name
    tokenized_vuln_name = tokenizer.texts_to_sequences([VULN_NAME])
    tokenized_vuln_name_padded = pad_sequences(tokenized_vuln_name, maxlen=20, padding='post')
    # Tokenizing Vuln Description
    tokenized_vuln_desc = tokenizer.texts_to_sequences([VULN_DESC])
    tokenized_vuln_desc_padded = pad_sequences(tokenized_vuln_desc, maxlen=800, padding='post')
    # Tokenizing severity
    tokenized_vuln_severity = tokenizer.texts_to_sequences([SEVERITY])
    tokenized_data_set = np.concatenate((tokenized_app_name_padded,
                                         tokenized_app_context_padded,
                                         tokenized_vuln_name_padded,
                                         tokenized_vuln_desc_padded,
                                         tokenized_vuln_severity),
                                        axis=1)
    return tokenized_data_set

def load_vuln_from_request_array(requests, tokenizer):
    for request in requests:
        APP_NAME = 'XXBOS XXAN ' + request.appName
        APP_CONTEXT = 'XXAC ' + request.appContext
        VULN_NAME = 'XXVN ' + request.vulnName
        VULN_DESC = 'XXVD ' + request.vulnDescription
        SEVERITY = 'XXSV ' + request.severity + ' XXEOS'
        # tokenizing app_name
        tokenized_app_name = tokenizer.texts_to_sequences([APP_NAME])
        tokenized_app_name_padded = pad_sequences(tokenized_app_name, maxlen=20, padding='post')
        # tokenizing app_context
        tokenized_app_context = tokenizer.texts_to_sequences([APP_CONTEXT])
        tokenized_app_context_padded = pad_sequences(tokenized_app_name, maxlen=20, padding='post')
        # Tokenizing vuln name
        tokenized_vuln_name = tokenizer.texts_to_sequences([VULN_NAME])
        tokenized_vuln_name_padded = pad_sequences(tokenized_vuln_name, maxlen=20, padding='post')
        # Tokenizing Vuln Description
        tokenized_vuln_desc = tokenizer.texts_to_sequences([VULN_DESC])
        tokenized_vuln_desc_padded = pad_sequences(tokenized_vuln_desc, maxlen=800, padding='post')
        # Tokenizing severity
        tokenized_vuln_severity = tokenizer.texts_to_sequences([SEVERITY])
        tokenized_data_set = np.concatenate((tokenized_app_name_padded,
                                             tokenized_app_context_padded,
                                             tokenized_vuln_name_padded,
                                             tokenized_vuln_desc_padded,
                                             tokenized_vuln_severity),
                                            axis=1)
    return tokenized_data_set
