# Pipeline de machine learning para deteccion de spam

import pandas as pd
import re
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# =========================
# FUNCION DE LIMPIEZA
# =========================

def clean_text(text):
    text = text.lower()                 # convertir a minúsculas
    text = re.sub(r'\d+', '', text)     # eliminar números
    text = re.sub(r'[^\w\s]', '', text) # eliminar signos de puntuación
    text = re.sub(r'\s+', ' ', text)    # eliminar espacios extra
    return text.strip()


# =========================
# CARGAR DATASET
# =========================

print("Cargando dataset...")

try:
    df = pd.read_csv("data/spam.csv", encoding="latin-1")

    if df.shape[1] == 1:
        df = pd.read_csv("data/spam.csv", encoding="latin-1", sep=";")

    if df.shape[1] == 1:
        df = pd.read_csv("data/spam.csv", encoding="latin-1", sep="\t")

except Exception as e:
    print("Error leyendo el CSV:", e)
    exit()

print("Columnas detectadas:", df.columns)
print("Shape:", df.shape)


# =========================
# NORMALIZAR COLUMNAS
# =========================

if "v1" in df.columns and "v2" in df.columns:
    df = df.rename(columns={"v1": "categoria", "v2": "mensaje"})

elif df.shape[1] >= 2:
    df = df.iloc[:, :2]
    df.columns = ["categoria", "mensaje"]

else:
    print("No se pudieron detectar columnas correctas")
    exit()

print("Columnas finales:", df.columns)


# =========================
# LIMPIAR TEXTO
# =========================

df["mensaje"] = df["mensaje"].apply(clean_text)


# =========================
# CONVERTIR ETIQUETAS
# =========================

df["categoria"] = df["categoria"].map({"ham": 0, "spam": 1})


# =========================
# ELIMINAR NULOS
# =========================

df = df.dropna(subset=["mensaje", "categoria"])


# =========================
# DIVIDIR DATOS
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    df["mensaje"],
    df["categoria"],
    test_size=0.2,
    random_state=42
)


# =========================
# VECTORIZAR TEXTO
# =========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,3),
    max_features=7000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =========================
# ENTRENAR MODELO
# =========================

model = MultinomialNB()
model.fit(X_train_vec, y_train)


# =========================
# EVALUAR MODELO
# =========================

y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy del modelo de spam: {acc:.4f}")

print("\nReporte de clasificación:")

print(classification_report(y_test, y_pred))


# =========================
# GUARDAR MODELO
# =========================

os.makedirs("src/model", exist_ok=True)

with open("src/model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("src/model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModelo y vectorizador guardados correctamente")