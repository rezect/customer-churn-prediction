sudo apt update
sudo apt install git
sudo git clone https://github.com/rezect/customer-churn-prediction
cd customer-churn-prediction

# Скачиваем модель
mkdir models
curl -L "https://drive.google.com/file/d/11QEBo9sArke_yEjV-TqpAscA8vAJlo1u/view?usp=drive_link" -o models/model.joblib

sudo apt install nginx

# Настраиваем конфиг nginx
cat default.conf > /etc/nginx/conf.d/default.conf
cat nginx.conf > /etc/nginx/nginx.conf
sudo mkdir -p /var/www/templates
sudo chown -R www-data:www-data /var/www/templates
cp templates/index.html /var/www/templates/index.html
# Установка SSL сертификата
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d <your domain>.ru -d www.<your domain>.ru


sudo apt install python3-pip
apt install python3.12-venv
python3 -m venv .venv
chmod +x .venv/bin/activate
sourse .venv/bin/activate
# Запускаем нашу АПИшку в фоне
nohup python3 src/app.py > app.log &

# Запускаем nginx
sudo systemctl start nginx
sudo systemctl enable nginx