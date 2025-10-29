from src.utils import download_telco_churn_dataset
import os
import gdown

model_url = "https://drive.google.com/file/d/11QEBo9sArke_yEjV-TqpAscA8vAJlo1u/view?usp=drive_link"

if __name__ == "__main__":
    # Загрузка датасета
    download_telco_churn_dataset()
    
    # Загрузка модели
    if (not os.path.exists("models")):
        os.mkdir('models')  
        
    model_path = "models/model.joblib"
    gdown.download(model_url, model_path, fuzzy=True)
    