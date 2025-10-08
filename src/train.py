from preprocessing import get_preproc
from utils import download_telco_churn_dataset, split_test_train
from models import *
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, classification_report
import pandas as pd

if __name__ == "__main__":
    telco = pd.read_csv("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = split_test_train(telco)
    
    preprocessing = get_preproc()
    
    # Тест первой модели на тренировочных данных
    xgb_pipeline = make_pipeline(preprocessing, xgb)
    
    xgb_errors = cross_val_score(xgb_pipeline, X_train, y_train, cv=10, scoring='f1', n_jobs=-1)
    
    print(pd.DataFrame(xgb_errors).describe())
    
    xgb_pipeline.fit(X_train, y_train)
    y_pred = xgb_pipeline.predict(X_test)
    
    # Calculate test score
    from sklearn.metrics import f1_score
    test_f1 = f1_score(y_test, y_pred)
    print(f"\nTest F1 Score: {test_f1:.4f}")
    
    # Baseline модель
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    print(f"\nBaseline F1 (most_frequent): {f1_score(y_test, y_pred_dummy):.4f}")
    
    # Ваша модель
    print("\n=== ОТЧЁТ ПО ВАШЕЙ МОДЕЛИ ===")
    print(classification_report(y_test, y_pred))
