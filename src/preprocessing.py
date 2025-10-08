from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, OrdinalEncoder
import pandas as pd
import numpy as np


def get_preproc():
    num_cols = ["MonthlyCharges", "tenure"]
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity",
                   "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]
    cat_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]

    class LabelEncoder_only_category_(BaseEstimator, TransformerMixin):
        def __init__(self, only_category):
            self.only_category = only_category
        
        def fit(self, X, y=None):
            # Просто сохраняем количество признаков
            if isinstance(X, pd.DataFrame):
                self.n_features_ = X.shape[1]
                self.feature_names_ = X.columns.tolist()
            else:
                self.n_features_ = X.shape[1] if len(X.shape) > 1 else 1
                self.feature_names_ = None
            return self
        
        def transform(self, X):
            # Преобразуем DataFrame в numpy если нужно
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            # Бинарное кодирование: 1 если значение == only_category, иначе 0
            X_binary = (X_array == self.only_category).astype(int)
            
            return X_binary
        
        def get_feature_names_out(self, input_features=None):
            if self.feature_names_ is not None:
                return np.array(self.feature_names_)
            elif input_features is not None:
                return np.array(input_features)
            else:
                return np.array([f"binary_{i}" for i in range(self.n_features_)])

    def to_numeric_(X: pd.DataFrame):
        """Приводит все записи в табличке к числовым значениям"""
        X = X.copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    default_num_pipeline = make_pipeline(
        FunctionTransformer(to_numeric_, feature_names_out="one-to-one"),
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )
    sqrt_num_pipeline = make_pipeline(
        FunctionTransformer(to_numeric_, feature_names_out="one-to-one"),
        SimpleImputer(strategy="mean"),
        FunctionTransformer(np.sqrt, feature_names_out="one-to-one"),
        StandardScaler()
    )
    
    yes_no_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        LabelEncoder_only_category_("Yes"),
    )
    
    onehot_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    return ColumnTransformer([
        # Drop useless columns
        ("drop", "drop", ["customerID"]),
        # Числовые признаки
        ("num", default_num_pipeline, num_cols),
        ("sqrt", sqrt_num_pipeline, ["TotalCharges"]),
        ("yes_no", yes_no_pipeline, yes_no_cols),
        ("1hot", onehot_pipeline, cat_cols),
    ], remainder='passthrough')
