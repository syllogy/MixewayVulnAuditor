from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix


def test_random_forest(data_train_sequence, labels_train, data_test_sequence, labels_test ):
    print("============== Random Forest =========================")
    # Create the model with 100 trees
    random_forest = RandomForestClassifier(n_estimators=100,
                                           bootstrap = True,
                                           max_features = 'auto')
    # Fit on training data
    random_forest.fit(data_train_sequence, labels_train)
    # Actual class predictions
    y_pred = random_forest.predict(data_test_sequence)
    print("Random Forest Accuracy: " + str(accuracy_score(labels_test,y_pred)))
    print("Random Forest Recall: " + str(recall_score(labels_test,y_pred)))
    print("Random Forrest Confusion matrix" + str(confusion_matrix(labels_test,y_pred)))