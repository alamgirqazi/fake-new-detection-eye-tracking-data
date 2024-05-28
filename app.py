import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score


def apply_xgb(X_train, y_train, X_test, y_test):

    xgb_predictions = train_xgboost(
        X_train, y_train, X_test, y_test)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    print(f'XGBoost Accuracy: {xgb_accuracy:.2f}')
    print("XGBoost Classification Report:")
    print(classification_report(y_test, xgb_predictions, zero_division=1))


# Here we train XGBoost classifier
def train_xgboost(X_train, y_train, X_test, y_test):
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = xgb_model.predict(X_test)

    return y_pred


def init():
    # dataset for all participants
    folder_path = 'dataset/entire/D3-Eye-movements-data'
    # dataset for single participant
    # folder_path = 'dataset/single-participant'

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    dataframes = []
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        condition_value = 1 if 'fake' in file_name else 0
        df['fake_news'] = condition_value
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)

    print(data)

    selected_features = data[[
        'meanPupilDiameter',
        'startSaccadeX',
        'startSaccadeY',
        'endSaccadeX',
        'endSaccadeY',
        'fake_news']]

    target_variable = selected_features['fake_news']

    fake_news_count = (data['fake_news'] == 1).sum()
    print("Count where fake NEWS : ", fake_news_count)

    real_news_count = (data['fake_news'] == 0).sum()
    print("Count where real NEWS : ", real_news_count)

    selected_features = selected_features.copy()

    selected_features.drop(columns=['fake_news'], inplace=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        selected_features, target_variable, test_size=0.2, random_state=42)

    apply_xgb(X_train, y_train, X_test, y_test)


init()
