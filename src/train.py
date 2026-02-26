# Pipeline de machine learning para deteccion de spam

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 📌 Cargar dataset (lectura robusta)
print("Cargando dataset...")

try:
    # Intento normal (coma)
    df = pd.read_csv("data/spam.csv", encoding="latin-1")
    
    # Si solo detecta 1 columna, probar con ;
    if df.shape[1] == 1:
        df = pd.read_csv("data/spam.csv", encoding="latin-1", sep=";")
    
    # Si aún hay 1 columna, probar con tab
    if df.shape[1] == 1:
        df = pd.read_csv("data/spam.csv", encoding="latin-1", sep="\t")

except Exception as e:
    print("❌ Error leyendo el CSV:", e)
    exit()

print("Columnas detectadas:", df.columns)
print("Shape:", df.shape)

# 🏷️ Normalizar nombres de columnas
if "v1" in df.columns and "v2" in df.columns:
    df = df.rename(columns={"v1": "label", "v2": "message"})
elif df.shape[1] >= 2:
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]
else:
    print("❌ No se pudieron detectar columnas correctas")
    exit()

print("Columnas finales:", df.columns)

# 🔢 Convertir etiquetas a números
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ⚠️ Eliminar filas con valores nulos
df = df.dropna(subset=["message", "label"])

# 📊 Separar datos
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# 🔤 Vectorizar texto
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🤖 Entrenar modelo
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 📈 Evaluar
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy del modelo: {acc:.4f}")