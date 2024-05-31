from helpers.train import train_svm, train_xgboost, apply_adaboost, train_decision_tree, train_knn, train_logistic_regression, train_random_forest, train_neural_network
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


def apply_knn(X_train, y_train, X_test, y_test):

    # Train and evaluate K Nearest Neighbors classifier
    knn_predictions = train_knn(
        X_train, y_train, X_test, y_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    print(f'K Nearest Neighbors Accuracy: {knn_accuracy:.2f}')
    print("K Nearest Neighbors Classification Report:")
    print(classification_report(y_test, knn_predictions, zero_division=1))
    # print("K Nearest Neighbors Confusion Matrix:")
    # print(confusion_matrix(y_test, knn_predictions))


def apply_logistic_regression(X_train, y_train, X_test, y_test):

    # Train and evaluate Logistic Regression classifier
    lr_predictions = train_logistic_regression(
        X_train, y_train, X_test, y_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print(f'Logistic Regression Accuracy: {lr_accuracy:.2f}')
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, lr_predictions, zero_division=1))
    # print("Logistic Regression Confusion Matrix:")
    # print(confusion_matrix(y_test, lr_predictions))


# def apply_xgb(X_train, y_train, X_test, y_test):

#     # Train and evaluate XGBoost classifier
#     xgb_predictions = train_xgboost(
#         X_train, y_train, X_test, y_test)
#     xgb_accuracy = accuracy_score(y_test, xgb_predictions)
#     print(f'XGBoost Accuracy: {xgb_accuracy:.2f}')
#     print("XGBoost Classification Report:")
#     print(classification_report(y_test, xgb_predictions, zero_division=1))
#     # print("XGBoost Confusion Matrix:")
#     # print(confusion_matrix(y_test, xgb_predictions))


def apply_svm(X_train, y_train, X_test, y_test):

    # Train and evaluate SVM classifier
    svm_predictions = train_svm(X_train, y_train, X_test, y_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print(f'SVM Accuracy: {svm_accuracy:.2f}')
    print("SVM Classification Report:")
    print(classification_report(y_test, svm_predictions, zero_division=1))
    # print("SVM Confusion Matrix:")
    # print(confusion_matrix(y_test, svm_predictions))


def apply_random_forest(X_train, y_train, X_test, y_test):
    # Train and evaluate Random Forest classifier
    rf_predictions = train_random_forest(
        X_train, y_train, X_test, y_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f'Random Forest Accuracy: {rf_accuracy:.2f}')
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_predictions))


def apply_nn(X_train, y_train, X_test, y_test):
    # Train and evaluate Neural Network classifier
    nn_predictions = train_neural_network(
        X_train, y_train, X_test, y_test)
    nn_accuracy = accuracy_score(y_test, nn_predictions)
    print(f'Neural Network Accuracy: {nn_accuracy:.2f}')
    print("Neural Network Classification Report:")
    print(classification_report(y_test, nn_predictions, zero_division=1))
    # print("Neural Network Confusion Matrix:")
    # print(confusion_matrix(y_test, nn_predictions))
    # Once the processing is done, print a newline to clear the loader animation
    print('\nProcessing completed!')


def apply_decision_tree(X_train, y_train, X_test, y_test):
    # Train and evaluate Decision Tree classifier
    decision_tree_predictions = train_decision_tree(
        X_train, y_train, X_test, y_test)
    decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)
    print(f'Decision Tree Accuracy: {decision_tree_accuracy:.2f}')
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, decision_tree_predictions, zero_division=1))
    # print("Decision Tree Confusion Matrix:")
    # print(confusion_matrix(y_test, decision_tree_predictions))
