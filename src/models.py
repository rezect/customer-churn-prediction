from sklearn.model_selection import RandomizedSearchCV
from utils import split_test_train
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump, load
import pandas as pd
import numpy as np
from preprocessing import get_preproc

# Model
from catboost import CatBoostClassifier

RND_SEED = 42


def fine_tuning_models(models_data, verbose=0):
    preprocessing = get_preproc()
    
    for name, model_data in models_data.items():
        models_data[name] = {
            "pipeline": ImbPipeline([
                ("preproc", preprocessing.named_steps['preproc']),
                ("smote", preprocessing.named_steps['smote']),
                ("model", model_data["model"]),
            ]),
            "param_distrib": model_data["param_disturb"],
        }

        roc_auc_losses = cross_val_score(
            models_data[name]["pipeline"], X_train, y_train, cv=10, n_jobs=-1, scoring='roc_auc')
        if (verbose == 1):
            print(f'{name} ROC-AUC:\n{pd.Series(roc_auc_losses).mean()}')

    if (verbose == 1):
        print('=' * 50 + "Tuned models!!!" + '=' * 50)

    best_models = {}
    n_iter = 100

    for name, model_data in models_data.items():
        rnd_search = RandomizedSearchCV(model_data["pipeline"], param_distributions=model_data["param_distrib"],
                                        n_iter=n_iter, cv=5, n_jobs=-1, random_state=RND_SEED, scoring='roc_auc')
        rnd_search.fit(X_train, y_train)

        roc_auc_losses_cv = rnd_search.best_score_

        if (verbose == 1):
            print(f'{name} AUC:\n{pd.Series(roc_auc_losses_cv).mean()}')

        best_models[name] = rnd_search.best_estimator_

    return best_models


def get_optimal_threshold(model, target='f1', target_score=0.8, help_metric='recall', help_metric_min_score=0.7):
    from sklearn.metrics import precision_recall_curve

    if (target not in ['f1', 'precision', 'recall']):
        raise ValueError("target must be 'f1', 'precision' or 'recall'.")

    if (help_metric not in ['precision', 'recall']):
        raise ValueError("target must be 'precision' or 'recall'.")

    def f1(precision_, recall_):
        if precision_ + recall_ == 0:
            return 0.0

        return 2 * (precision_ * recall_) / (precision_ + recall_)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_test, y_pred_proba)

    f1s = []
    max_f1_score = 0
    max_f1_idx = 0

    for i in range(len(thresholds)):
        f1s.append(f1(precision[i], recall[i]))
        if max_f1_score < f1s[i]:
            if (help_metric is None):
                max_f1_score = f1s[i]
                max_f1_idx = i
            elif (help_metric == 'recall') and (recall[i] >= help_metric_min_score):
                max_f1_score = f1s[i]
                max_f1_idx = i
            elif (help_metric == 'precision') and (precision[i] >= help_metric_min_score):
                max_f1_score = f1s[i]
                max_f1_idx = i

    optimal_idx_recall = np.argmin(recall >= target_score)
    optimal_idx_precision = np.argmin(recall >= target_score)
    optimal_threshold_recall = thresholds[optimal_idx_recall]
    optimal_threshold_precision = thresholds[optimal_idx_precision]
    optimal_threshold_f1 = thresholds[max_f1_idx]

    if target == 'f1':
        return optimal_threshold_f1
    elif target == 'recall':
        return optimal_threshold_recall
    elif target == 'precision':
        return optimal_threshold_precision


if __name__ == "__main__":
    telco = pd.read_csv("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = split_test_train(telco)

    catboost = CatBoostClassifier(verbose=0)

    # Создадим сетку для поиска лучшей модели
    pd_catboost = {
        'model__iterations': [100, 200, 500, 1000],
        'model__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'model__depth': [4, 6, 8, 10],
        'model__l2_leaf_reg': [1, 3, 5, 7, 9],
        'model__random_strength': [0.5, 1, 2, 5],
        'model__bagging_temperature': [0, 0.5, 1, 2],
        'model__border_count': [32, 64, 128, 254],
        'model__min_data_in_leaf': [1, 3, 5, 10, 20]
    }

    models_data = {
        "CatBoost": {
            "model": catboost,
            "param_disturb": pd_catboost,
        },
    }

    print("Fine Tuning Started!\n")
    best_models = fine_tuning_models(models_data)
    catboost_tuned = best_models["CatBoost"]
    print("Fine Tuning Ended\n")

    from sklearn.model_selection import FixedThresholdClassifier

    # Тюнингуем threshold
    optimal_threshold_LGBM = get_optimal_threshold(
        model=catboost_tuned, target='f1', help_metric='recall', help_metric_min_score=0.8)
    catboost_tuned = FixedThresholdClassifier(
        catboost_tuned, threshold=optimal_threshold_LGBM)

    # Сохраняем модель
    model_path = 'models/model.joblib'
    dump(catboost_tuned, model_path)
    model = load(model_path)

    print(f1_score(y_test, model.predict(X_test)))
