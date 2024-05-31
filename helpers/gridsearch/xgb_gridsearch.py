
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import xgboost as xgb

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [20, 50, 100, 200, 500],
    'max_depth': [3, 4, 5,  7, 10],
    # 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'min_child_weight': [1, 3, 5, 7],
    # 'gamma': [0, 0.1, 0.3, 0.5, 1],
    # 'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    # 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    # 'scale_pos_weight': [1, 10, 25, 50, 75, 99],
    # 'reg_alpha': [0, 0.01, 0.1, 1],
    # 'reg_lambda': [0, 0.01, 0.1, 1]
}


def apply_xgb_gridsearch(X_train_scaled, y_train):

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               scoring='accuracy', n_jobs=-1, cv=5, verbose=2)

    grid_search.fit(X_train_scaled, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best accuracy found: ", grid_search.best_score_)
