from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
import numpy as np


class LabelEncoder_only_category_(BaseEstimator, TransformerMixin):
    def __init__(self, only_category):
        self.only_category = only_category

    def fit(self, X, y=None, sample_weight=None):
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
        return input_features

class FeaturesEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None, drop_patterns=None):
        # Используем кортежи для immutable параметров
        self.columns_to_drop = tuple(columns_to_drop) if columns_to_drop is not None else ()
        self.drop_patterns = tuple(drop_patterns) if drop_patterns is not None else ()
    
    def fit(self, X, y=None, sample_weight=None):
        return self
        
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        
        # Вычисляем колонки для удаления каждый раз
        columns_to_drop = []
        
        # Добавляем явно указанные колонки
        columns_to_drop.extend(self.columns_to_drop)
        
        # Добавляем колонки по паттернам
        for pattern in self.drop_patterns:
            matching_cols = [col for col in X.columns if pattern in col]
            columns_to_drop.extend(matching_cols)
        
        # Убираем дубликаты и оставляем только существующие колонки
        existing_cols_to_drop = list(set(
            col for col in columns_to_drop if col in X.columns
        ))
        
        X = X.drop(columns=existing_cols_to_drop)
        return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        
        # Вычисляем колонки для удаления
        columns_to_drop = []
        columns_to_drop.extend(self.columns_to_drop)
        
        for pattern in self.drop_patterns:
            matching_cols = [col for col in input_features if pattern in col]
            columns_to_drop.extend(matching_cols)
        
        columns_to_drop = set(columns_to_drop)
        remaining_features = [feat for feat in input_features 
                            if feat not in columns_to_drop]
        return np.array(remaining_features)

def to_numeric_(X: pd.DataFrame):
    """Приводит все записи в табличке к числовым значениям"""
    X = X.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X

def get_preproc():
    num_cols = ["MonthlyCharges", "tenure"]
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity",
                   "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]
    cat_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]

    default_num_pipeline = make_pipeline(
        FunctionTransformer(
            to_numeric_, feature_names_out="one-to-one", validate=False),
        SimpleImputer(strategy="mean"),
        PolynomialFeatures(degree=2, include_bias=False,
                           interaction_only=True),
        StandardScaler(),
    )
    sqrt_num_pipeline = make_pipeline(
        FunctionTransformer(
            to_numeric_, feature_names_out="one-to-one", validate=False),
        SimpleImputer(strategy="mean"),
        FunctionTransformer(
            np.sqrt, feature_names_out="one-to-one", validate=False),
        StandardScaler()
    )

    yes_no_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        LabelEncoder_only_category_("Yes"),
    )

    onehot_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )

    preprocessing = ColumnTransformer([
        ("drop", "drop", ["customerID"]),
        ("num", default_num_pipeline, num_cols),
        ("sqrt", sqrt_num_pipeline, ["TotalCharges"]),
        ("yes_no", yes_no_pipeline, yes_no_cols),
        ("1hot", onehot_pipeline, cat_cols),
    ], remainder='passthrough').set_output(transform='pandas')

    full_pipeline = ImbPipeline([
        ("preproc", preprocessing),
        ("drop", FeaturesEngineer(columns_to_drop=["1hot__gender_Female", "1hot__gender_Male", "yes_no__StreamingMovies", "yes_no__MultipleLines",
         "yes_no__PhoneService", "yes_no__DeviceProtection", "yes_no__OnlineBackup", "1hot__PaymentMethod_Mailed check", "poly_features__1"])),
        ("smote", SMOTE(
            k_neighbors=5,
            random_state=42,
            sampling_strategy='auto',
        )),
    ]).set_output(transform='pandas')

    return full_pipeline


if __name__ == "__main__":
    telco = pd.read_csv("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    preprocessing = get_preproc()
    
    # Сохраняем пайплайн в файл
    from joblib import dump, load
    
    dump(preprocessing, 'models/preprocessing.joblib')
    preprocessing = load('models/preprocessing.joblib')
    
    telco['Churn'] = telco['Churn'].map({'Yes': 1, 'No': 0})

    y = telco["Churn"]
    X = telco.drop(columns="Churn")

    X, y = preprocessing.fit_resample(X, y)

    print(len(preprocessing.get_feature_names_out()))

