from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


def train_svm(X_train_scaled, y_train, X_test_scaled, y_test):

    svm_model = SVC(kernel='linear', C=0.1)
    svm_model.fit(X_train_scaled, y_train)
    # Make predictions on the test set
    y_pred = svm_model.predict(X_test_scaled)

    return y_pred


def train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test):
    # Train XGBoost classifier
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = xgb_model.predict(X_test_scaled)

    return y_pred


def train_adaboost(X_train_scaled, y_train, X_test_scaled, y_test):
    # Train AdaBoost classifier
    adaboost_model = AdaBoostClassifier(n_estimators=100, random_state=42)
    adaboost_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = adaboost_model.predict(X_test_scaled)

    return y_pred


def train_decision_tree(X_train_scaled, y_train, X_test_scaled, y_test):
    # Train Decision Tree classifier
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = dt_model.predict(X_test_scaled)

    return y_pred


def train_knn(X_train_scaled, y_train, X_test_scaled, y_test):
    # Train K Nearest Neighbors classifier
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = knn_model.predict(X_test_scaled)

    return y_pred

# Define the function to train Logistic Regression model


def train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test):
    # Train Logistic Regression classifier
    lr_model = LogisticRegression()
    lr_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = lr_model.predict(X_test_scaled)

    return y_pred


def train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test):
    # Train Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=12, random_state=42, max_depth=4)
    rf_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test_scaled)

    return y_pred


def train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test):
    # Convert labels to categorical one-hot encoding
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)

    # Define the neural network model
    model = Sequential()
    model.add(Dense(64, activation='relu',
              input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_train_categorical.shape[1], activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_scaled, y_train_categorical, epochs=50,
              batch_size=64, validation_data=(X_test_scaled, y_test_categorical))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(
        X_test_scaled, y_test_categorical)
    print(f'Neural Network Test Accuracy: {test_accuracy:.2f}')

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = tf.argmax(y_pred, axis=1).numpy()

    return y_pred_classes


def apply_adaboost(X_train_scaled, y_train, X_test_scaled, y_test):
    adaboost_predictions = train_adaboost(
        X_train_scaled, y_train, X_test_scaled, y_test)
    adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)
    print(f'AdaBoost Accuracy: {adaboost_accuracy:.2f}')
    print("AdaBoost Classification Report:")
    print(classification_report(y_test, adaboost_predictions, zero_division=1))
    # print("AdaBoost Confusion Matrix:")
    # print(confusion_matrix(y_test, adaboost_predictions))
