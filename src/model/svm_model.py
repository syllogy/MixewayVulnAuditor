from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import time


def build_svm_model(train_dataset,train_labels,test_dataset,test_labels):
    print("======================== SVM ============================")
    svclassifier = SVC(kernel='poly', degree=8)
    start_time = time.time()
    svclassifier.fit(train_dataset, train_labels)
    print("--- %s Training Time ---" % (time.time() - start_time))
    y_pred = svclassifier.predict(test_dataset)
    print(confusion_matrix(test_dataset, test_labels))
    print(classification_report(test_dataset, test_labels))