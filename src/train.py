# Mejora Sprint 6: limpieza avanzada para español

import pandas as pd
import re
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# LIMPIEZA DE TEXTO
# =========================
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# =========================
# CARGAR DATASET
# =========================
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# normalizar columnas
if "v1" in df.columns:
    df = df.rename(columns={"v1": "categoria", "v2": "mensaje"})
else:
    df = df.iloc[:, :2]
    df.columns = ["categoria", "mensaje"]

# limpiar texto
df["mensaje"] = df["mensaje"].astype(str).apply(limpiar_texto)

# mapear etiquetas
df["categoria"] = df["categoria"].map({"ham": 0, "spam": 1})

df = df.dropna()

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["mensaje"],
    df["categoria"],
    test_size=0.2,
    random_state=42
)

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# MODELO
# =========================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# =========================
# MÉTRICAS
# =========================
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte:")
print(classification_report(y_test, y_pred))
print("\nMatriz:")
print(confusion_matrix(y_test, y_pred))

# =========================
# GUARDAR
# =========================
os.makedirs("src/model", exist_ok=True)

with open("src/model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("src/model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Modelo guardado correctamente")