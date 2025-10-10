from preprocessing import get_preproc
from utils import download_telco_churn_dataset, split_test_train
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils import class_weight
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform, randint

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

# Utils
import pandas as pd
import sys

sys.path.append('C:\\Coding\\customer-churn-prediction\\src')

RND_SEED = 42

telco = pd.read_csv("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
X_train, X_test, y_train, y_test = split_test_train(telco)

preprocessing = get_preproc()


def fit_and_evaluate(models, cv=10):
    # Крафтим пайплайны для каждой модели
    pipelines = {}

    for name, model in models:
        pipelines[name] = (Pipeline([
            ("preproc", preprocessing),
            ("model", model)
        ]))

    metrics = {}

    # Оценивать модели будем по f1 score, но важна нам именно метрика recall для положительного класса
    for name, model in pipelines.items():
        # Оценим сырые модельки на CV
        roc_auc_train = cross_val_score(
            model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1).mean()
        f1_cv = cross_val_score(
            model, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1).mean()

        # Исправляем дисбаланс классов
        classes_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )
        model.fit(X_train, y_train, model__sample_weight=classes_weights)

        # print(f"TRAIN ROC-AUC: {roc_auc_train}")
        y_pred_test = model.predict(X_test)

        # Metrics evaluating
        roc_auc_test = roc_auc_score(y_test, y_pred_test)
        # print(f"TEST ROC-AUC: {roc_auc_test}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        cm_df = pd.DataFrame(cm,
                             index=['Факт: 0', 'Факт: 1'],
                             columns=['Прогноз: 0', 'Прогноз: 1'])
        # print(cm_df)

        metrics[name] = {
            'roc_auc_test': roc_auc_test,
            'roc_auc_cv': roc_auc_train,
            'f1_cv': f1_cv,
        }

    sorted_metrics = dict(
        sorted(metrics.items(), key=lambda item: -item[1]['f1_cv']))

    print(f"ТОП МОДЕЛЕЙ:")
    for name, metric in sorted_metrics.items():
        print(
            "\n" + f"{name} : \nF1: {metric['f1_cv']}\nAUC: {metric['roc_auc_cv']}")


def fine_tuning_models(models_data):
    for name, model_data in models_data.items():
        models_data[name] = {
            "pipeline": Pipeline([
                ("preproc", preprocessing),
                ("model", model_data["model"]),
            ]),
            "param_distrib": model_data["param_disturb"],
        }

        f1_losses = cross_val_score(
            models_data[name]["pipeline"], X_train, y_train, cv=10, n_jobs=-1, scoring='f1')
        print(f'{name} f1:\n{pd.Series(f1_losses).mean()}')

    # Исправляем дисбаланс классов
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    print('=' * 50 + "Tuned models!!!" + '=' * 50)

    best_models = {}

    for name, model_data in models_data.items():
        rnd_search = RandomizedSearchCV(model_data["pipeline"], param_distributions=model_data["param_distrib"],
                                        n_iter=50, cv=5, n_jobs=-1, random_state=RND_SEED, scoring='f1')
        rnd_search.fit(X_train, y_train, model__sample_weight=classes_weights)

        f1_losses_cv = rnd_search.best_score_

        print(f'{name} f1:\n{pd.Series(f1_losses_cv).mean()}')

        best_models[name] = rnd_search.best_estimator_

    return best_models


if __name__ == "__main__":

    # Создаем сырые модели для проверки
    lgbm = LGBMClassifier(random_state=RND_SEED, verbose=-1)
    lr = LogisticRegression(random_state=RND_SEED, penalty='l2')
    lin_svc = LinearSVC(C=1, random_state=RND_SEED)
    gb = GradientBoostingClassifier(random_state=RND_SEED)
    svc = SVC(C=1, random_state=RND_SEED)
    rnd_forest = RandomForestClassifier(random_state=RND_SEED, n_jobs=-1)
    xgb = XGBClassifier(random_state=RND_SEED, n_jobs=-1)

    models = [
        ("Logistic Regression L2", lr),
        ("LightGBM", lgbm),
        ("RND Forest", rnd_forest),
        ("XGB Classifier", xgb),
        ("Gradient Boosting", gb),
        ("Linear SVC", lin_svc),
        ("SVC", svc),
    ]

    # Находим топ лучших моделей по f1 score
    fit_and_evaluate(models)

    # Задаем сетки для поиска лучших параметров 3х лучших моделей моделей
    pd_lin_svc = {
        "model__tol": uniform(1e-6, 1e-4),
        "model__C": loguniform(0.1, 10),
        "model__fit_intercept": [True, False],
        "model__intercept_scaling": loguniform(1, 10),
    }

    pd_gb = {
        "model__learning_rate": loguniform(0.05, 0.1),
        "model__n_estimators": randint(100, 200),
        "model__max_depth": randint(3, 5),
        "model__max_features": ['sqrt', 'log2', None],
    }

    pd_lr = [
        {
            "model__penalty": ['l2'],
            "model__tol": uniform(1e-6, 1e-4),
            "model__C": loguniform(0.5, 10),
            "model__max_iter": randint(50, 500),
            "model__solver": ['lbfgs'],
        },
        {
            "model__penalty": ['l2', 'l1'],
            "model__tol": uniform(1e-6, 1e-4),
            "model__C": loguniform(0.5, 10),
            "model__max_iter": randint(50, 500),
            "model__solver": ['liblinear'],
        }
    ]

    models_data = {
        "Logistic Regression": {
            "model": lr,
            "param_disturb": pd_lr,
        },
        "Gradient Boosting": {
            "model": gb,
            "param_disturb": pd_gb,
        },
        "Linear SVC": {
            "model": lin_svc,
            "param_disturb": pd_lin_svc,
        },
    }
    
    best_models = fine_tuning_models(models_data)
    
    