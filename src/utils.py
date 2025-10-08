import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

RND_SEED = 42

def download_telco_churn_dataset(data_dir='../data/raw'):
    """
    Скачивает датасет Telco Customer Churn с Kaggle
    """
    # Создаем папку для данных, если её нет
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Инициализируем Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Скачиваем датасет
        dataset_name = 'blastchar/telco-customer-churn'
        api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
        
        print(f"✅ Датaсет успешно скачан в папку: {data_dir}")
        
        # Проверяем скачанные файлы
        files = os.listdir(data_dir)
        print(f"📁 Скачанные файлы: {files}")
        
        # Загружаем данные для проверки
        csv_file = [f for f in files if f.endswith('.csv')][0]
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        print(f"📊 Размер датасета: {df.shape}")
        print(f"🎯 Целевая переменная 'Churn': {df['Churn'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Ошибка при скачивании: {e}")
        return None
    
def split_test_train(df):
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop('Churn', axis=1)
    
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RND_SEED, stratify=y
    )
    
    return (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    df = download_telco_churn_dataset()