# Spam Detector ML

Proyecto de machine learning para detectar mensajes spam usando Naive Bayes y TF-IDF.

## Características

- Clasificación de mensajes spam / ham
- Vectorización con TF-IDF
- Modelo Multinomial Naive Bayes
- Evaluación con accuracy

## Instalación

1. Clonar repositorio:

git clone <URL_DEL_REPO>

2. Entrar al proyecto:

cd spam-detector-ml

3. Crear entorno virtual:

python -m venv venv

4. Activar entorno:

Windows:
venv\Scripts\activate

5. Instalar dependencias:

pip install -r requirements.txt

## Entrenar modelo

python src/train.py

## Probar predicción

python src/predict.py