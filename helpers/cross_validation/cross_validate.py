
def apply_cross_val(classifier, X, y, cv=5):
    scores = cross_val_score(classifier, X, y, cv=cv)
    print(f"Classifier: {classifier.__class__.__name__}")
    print(f"Cross-validation scores: {scores}")
    print(f"Mean score: {scores.mean()}\n")


# List of classifiers to evaluate
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    LogisticRegression(max_iter=10000),  # Increase max_iter for convergence
    xgb.XGBClassifier(),
    SVC(),
    RandomForestClassifier(),
    AdaBoostClassifier()
]

# Apply cross-validation for each classifier
for classifier in classifiers:
    apply_cross_val(classifier, selected_features, target_variable)
