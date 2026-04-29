 #HEAD
# Mejora Sprint 6: limpieza avanzada para español
# actualización
 #484de46 (feat: integración del modelo con aplicación web en Flask)
import pandas as pd
import re
import nltk
import joblib
import unicodedata

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Descargar recursos de nltk
nltk.download('stopwords')

# Stopwords y stemming en español
stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')

# ==========================
# FUNCIÓN DE LIMPIEZA
# ==========================
def limpiar_texto(texto):
    texto = texto.lower()

    # Eliminar acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')

    # Eliminar links
    texto = re.sub(r'http\S+|www\S+', '', texto)

    # Eliminar correos
    texto = re.sub(r'\S+@\S+', '', texto)

    # Eliminar números
    texto = re.sub(r'\d+', '', texto)

    # Eliminar caracteres especiales
    texto = re.sub(r'[^a-zA-Zñáéíóúü\s]', '', texto)

    palabras = texto.split()

    # Quitar stopwords y aplicar stemming
    palabras_limpias = []
    for palabra in palabras:
        if palabra not in stop_words and len(palabra) > 2:
            palabra = stemmer.stem(palabra)
            palabras_limpias.append(palabra)

    return " ".join(palabras_limpias)

# ==========================
# CARGAR DATASET
# ==========================
data = pd.read_csv("data/spam.csv")

# Cambia estos nombres si tu dataset usa otros
data = data.rename(columns={
    'label': 'etiqueta',
    'text': 'mensaje'
})

# Convertir etiquetas
data['etiqueta'] = data['etiqueta'].map({
    'spam': 1,
    'ham': 0
})

# Limpiar mensajes
data['mensaje'] = data['mensaje'].astype(str).apply(limpiar_texto)

# Eliminar filas vacías
data = data[data['mensaje'].str.strip() != '']

# ==========================
# TF-IDF MEJORADO
# ==========================
vectorizador = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X = vectorizador.fit_transform(data['mensaje'])
y = data['etiqueta']

# ==========================
# TRAIN / TEST
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# MODELO 1: NAIVE BAYES
# ==========================
modelo_nb = MultinomialNB()
modelo_nb.fit(X_train, y_train)

pred_nb = modelo_nb.predict(X_test)

print("\n=== NAIVE BAYES ===")
print("Accuracy:", accuracy_score(y_test, pred_nb))
print(classification_report(y_test, pred_nb))
print(confusion_matrix(y_test, pred_nb))

# ==========================
# MODELO 2: REGRESIÓN LOGÍSTICA
# ==========================
modelo_lr = LogisticRegression(max_iter=1000)
modelo_lr.fit(X_train, y_train)

pred_lr = modelo_lr.predict(X_test)

print("\n=== REGRESIÓN LOGÍSTICA ===")
print("Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))
print(confusion_matrix(y_test, pred_lr))

# ==========================
# GUARDAR EL MEJOR MODELO
# ==========================
accuracy_nb = accuracy_score(y_test, pred_nb)
accuracy_lr = accuracy_score(y_test, pred_lr)

if accuracy_lr > accuracy_nb:
    mejor_modelo = modelo_lr
    nombre_modelo = "Regresión Logística"
else:
    mejor_modelo = modelo_nb
    nombre_modelo = "Naive Bayes"

joblib.dump((mejor_modelo, vectorizador), "app/model.pkl")

print(f"\nMejor modelo guardado: {nombre_modelo}")