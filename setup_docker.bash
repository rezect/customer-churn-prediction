curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
sudo docker --version

sudo apt install git
git --version
git clone https://github.com/rezect/customer-churn-prediction

cd customer-churn-prediction/

mkdir models
curl -L "https://drive.google.com/file/d/11QEBo9sArke_yEjV-TqpAscA8vAJlo1u/view?usp=drive_link" -o models/model.joblib

docker compose up