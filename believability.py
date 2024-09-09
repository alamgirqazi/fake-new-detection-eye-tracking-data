import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# Define function to determine perceived truth

# Function to evaluate the model for each participant


def evaluate_model(model, X_train, X_test, y_train, y_test):
    pipeline = make_pipeline(StandardScaler(), model)

    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    cv_predictions = cross_val_predict(pipeline, X_train, y_train, cv=5)

    # Train the classifier on the entire training set
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    cv_conf_matrix = confusion_matrix(y_train, cv_predictions)
    cv_class_report = classification_report(y_train, cv_predictions)
    # print(cv_conf_matrix)
    # print(cv_class_report)
    # Evaluate the classifier on test set
    accuracy = accuracy_score(y_train, cv_predictions)
    f1 = f1_score(y_train, cv_predictions)
    conf_matrix = confusion_matrix(y_train, cv_predictions)
    # print(conf_matrix)
    return accuracy, f1, conf_matrix


def is_perceived(row):
    if row['version'] == 'fake' and row['believability'] in [1, 2]:
        return 0
    elif row['version'] == 'true' and row['believability'] in [4, 5]:
        return 1
    elif row['version'] == 'fake':
        return 1
    return 0


def process_csv(df):
    # Convert believability to integer type
    df['believability'] = df['believability'].astype(int)

    # Filter participants
    participants_to_filter = [23, 20, 13, 10, 9, 1]
    df = df.copy()

    df = df[df['participant'].isin(participants_to_filter)]

    # Add new columns based on conditions
    # df['is_perceived_true'] = df.apply(is_perceived, axis=1)
    df.loc[:, 'is_perceived_true'] = df.apply(is_perceived, axis=1)

    # df['is_real'] = np.where(df['version'] == 'fake', 0, 1)

    # Drop unnecessary columns
    deleteArray = ['version', 'is_perceived_true']

    # To store accuracy and F1 score per participant
    results = []

    # Iterate over each participant's data
    for participant, participant_df in df.groupby('participant'):
        print(f"\nProcessing participant {participant}...")

        # Prepare data for the current participant
        y = participant_df['is_perceived_true']
        X = participant_df.drop(deleteArray, axis=1)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Initialize classifier
        xgboost = XGBClassifier(eval_metric='logloss')

        # Evaluate the model for the current participant
        accuracy, f1, conf_matrix = evaluate_model(
            xgboost, X_train, X_test, y_train, y_test)
        print('accuracy - >', accuracy)
        # print(f1)
        # Store results for each participant
        results.append([participant, accuracy, f1, conf_matrix])

    # Convert results to DataFrame for easy manipulation
    results_df = pd.DataFrame(
        results, columns=['participant', 'accuracy', 'f1_score', 'conf_matrix'])

    # Plot bar chart for accuracy and F1 scores
    plot_accuracy_f1_chart(results_df)

    # Plot stacked bar chart for confusion matrix
    plot_confusion_matrix_chart(results_df)


# Function to plot the accuracy and F1 score chart
def plot_accuracy_f1_chart(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = np.arange(len(results_df['participant']))

    ax.bar(index, results_df['accuracy'],
           bar_width, label='Accuracy', color='b')

    ax.bar(index + bar_width,
           results_df['f1_score'], bar_width, label='F1 Score', color='g')

    ax.set_xlabel('Participant')
    ax.set_ylabel('Scores')
    ax.set_title('Accuracy and F1 Score by Participant')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(results_df['participant'])
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_chart(results_df):
    fig, ax = plt.subplots(figsize=(12, 7))

    bar_width = 0.6
    index = np.arange(len(results_df['participant']))

    # Define colors
    colors = {
        'True Positive': '#2ca02c',  # Green
        'False Positive': '#ff7f0e',  # Orange
        'False Negative': '#d62728',  # Blue
        'True Negative': '#1f77b4'  # Red
    }

    # Create stacked bar chart for confusion matrix
    for i, participant in enumerate(results_df['participant']):
        conf_matrix = results_df['conf_matrix'].iloc[i]
        bottom = 0
        ax.bar(index[i], conf_matrix[1][1], bar_width, color=colors['True Positive'],
               edgecolor='black', label='True Positive' if i == 0 else "")
        bottom += conf_matrix[1][1]
        ax.bar(index[i], conf_matrix[1][0], bar_width, bottom=bottom, color=colors['False Negative'],
               edgecolor='black', label='False Negative' if i == 0 else "")
        bottom += conf_matrix[1][0]
        ax.bar(index[i], conf_matrix[0][1], bar_width, bottom=bottom, color=colors['False Positive'],
               edgecolor='black', label='False Positive' if i == 0 else "")
        bottom += conf_matrix[0][1]
        ax.bar(index[i], conf_matrix[0][0], bar_width, bottom=bottom, color=colors['True Negative'],
               edgecolor='black', label='True Negative' if i == 0 else "")

    ax.set_xlabel('Participant')
    ax.set_ylabel('Count')
    ax.set_title('Confusion Matrix by Participant')
    ax.set_xticks(index)
    ax.set_xticklabels(results_df['participant'])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=4, title='Confusion Matrix Components')

    plt.tight_layout()
    plt.show()


believability_data = pd.read_csv(
    'dataset/entire/processed/D2-Processed-features.csv')
process_csv(believability_data)
