<img src="https://mixeway.io/wp-content/uploads/2020/07/vuln_auditor.png">

# Mixeway Vuln Auditor
Mixeway Vuln Auditor is application which is ment to:
1. Create test suite for Various Machine Learning algorithms to verify possibility to build efficient
Software Vulnerability Classifier
2. Create API to perdict a class (CRV, DNRV) of vulnerability

### Directory description
- model: contain saved model and tokenizer dictionary to be used by REST API
- plot: contain plots from generated test suites
- src: contain source code of prototype

# Test suite
Vuln Auditor contains implementation of test suite for Neural Network, Random Forrest and Support Vector Machine.

### Requirements
1. Directory /data/ has to contain `CSV` files with headers of ("app_name","app_context","vuln_name","vuln_desc","severity","grade").
Test and train data is not provided
2. TensorFlow 2.0+
3. Python 3 

### Usage
`py ./src/main/MixewayVulnAuditor.py`
 
 Example Output:
 ```shell script
Loading file:  C:\Users\gsiew\IdeaProjects\MixewayVulnAuditor\data\audit_code.csv
Loading file:  C:\Users\gsiew\IdeaProjects\MixewayVulnAuditor\data\audit_infra.csv
Loading file:  C:\Users\gsiew\IdeaProjects\MixewayVulnAuditor\data\audit_os.csv
Loading file:  C:\Users\gsiew\IdeaProjects\MixewayVulnAuditor\data\audit_webapp.csv
Prepared dictionary for tokenizer
Train set size:  37737
Train set Labels:  {0: 24254, 1: 13483}
Test set size:  16173
Test set Labels:  {0: 10384, 1: 5789}
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 863, 32)           607264    
_________________________________________________________________
flatten (Flatten)            (None, 27616)             0         
_________________________________________________________________
dense (Dense)                (None, 32)                883744    
_________________________________________________________________
dense_1 (Dense)              (None, 32)                1056      
_________________________________________________________________
dense_2 (Dense)              (None, 16)                528       
_________________________________________________________________
dense_3 (Dense)              (None, 16)                272       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 17        
=================================================================
Total params: 1,492,881
Trainable params: 1,492,881
Non-trainable params: 0
_________________________________________________________________
Layers:  7
Epoch 1/50
1180/1180 [==============================] - 17s 14ms/step - loss: 0.2768 - accuracy: 0.9699 - f1: 0.9494 - precision: 0.9511 - recall: 0.9554 - val_loss: 0.1249 - val_accuracy: 0.9850 - val_f1: 0.9779 - val_precision: 0.9644 - val_recall: 0.9940
Epoch 2/50
```
Plots with model metrics will be stored in `/plot/` directory

# Rest API

### Requirements:
- SSL Certificate and KEY avaliable location store in environment variable (`CERTIFICATE` and `PRIVATEKEY`)
- model directory has to contain model description, if it will be empty model would be created using test data in data directory

### Usage
`py ./src/main/vuln_auditor_server.py`

### REST API

Method `POST http://localhost:8445/vuln/perdict`

example body:
```json
[
	{
		"id": 1,
		"appName": "Apollo",
		"appContext": "type opensource customer internal",
		"vulnName": "CVE-2016-1000339",
		"vulnDescription": "In the Bouncy Castle JCE Provider version 1.55 and earlier the primary engine class used for AES was AESFastEngine. Due to the highly table driven approach used in the algorithm it turns out that if the data channel on the CPU can be monitored the lookup table accesses are sufficient to leak information on the AES key being used. There was also a leak in AESEngine although it was substantially less. XXEOS has been modified to remove any signs of leakage (testing carried out on Intel X86-64) and is now the primary AES class for the BC JCE provider from 1.56. Use of AESFastEngine is now only recommended where otherwise deemed appropriate.",
		"severity": "Medium"
	}
]
```

example output
```json
[
  {
    "id": 1,
    "audit": 0
  }
]
```



