import pandas as pd
import re
import nltk
import joblib
import unicodedata
import os

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# ==========================
# DESCARGAR RECURSOS
# ==========================
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')

# ==========================
# LIMPIEZA DE TEXTO
# ==========================
def limpiar_texto(texto):
    texto = str(texto).lower()

    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'\S+@\S+', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)

    palabras = texto.split()

    palabras_limpias = [
        stemmer.stem(p)
        for p in palabras
        if p not in stop_words and len(p) > 2
    ]

    return " ".join(palabras_limpias)

# ==========================
# CARGAR DATASET
# ==========================
data = pd.read_csv("data/spam.csv", encoding='utf-8-sig', on_bad_lines='skip')
data.columns = data.columns.str.strip()

print("Columnas:", data.columns)

data = data.rename(columns={
    'label': 'etiqueta',
    'message': 'mensaje'
})

data = data.dropna(subset=['mensaje', 'etiqueta'])

data['etiqueta'] = data['etiqueta'].astype(str).str.lower().str.strip()
data['etiqueta'] = data['etiqueta'].map({
    'spam': 1,
    'ham': 0
})

data = data.dropna(subset=['etiqueta'])

# ==========================
# LIMPIAR TEXTO
# ==========================
data['mensaje'] = data['mensaje'].apply(limpiar_texto)
data = data[data['mensaje'].str.strip() != '']

# ==========================
# BALANCEAR DATOS
# ==========================
spam = data[data['etiqueta'] == 1]
ham = data[data['etiqueta'] == 0]

spam_upsampled = resample(
    spam,
    replace=True,
    n_samples=len(ham),
    random_state=42
)

data = pd.concat([ham, spam_upsampled])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================
# TF-IDF
# ==========================
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.9
)

X = vectorizer.fit_transform(data['mensaje'])
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
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

pred_nb = model_nb.predict(X_test)

print("\n=== NAIVE BAYES ===")
print("Accuracy:", accuracy_score(y_test, pred_nb))
print(classification_report(y_test, pred_nb))

# ==========================
# MODELO 2: REGRESIÓN LOGÍSTICA
# ==========================
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

pred_lr = model_lr.predict(X_test)

print("\n=== REGRESIÓN LOGÍSTICA ===")
print("Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

# ==========================
# COMPARACIÓN
# ==========================
accuracy_nb = accuracy_score(y_test, pred_nb)
accuracy_lr = accuracy_score(y_test, pred_lr)

if accuracy_lr > accuracy_nb:
    model = model_lr
    nombre_modelo = "Regresión Logística"
else:
    model = model_nb
    nombre_modelo = "Naive Bayes"

print(f"\nMejor modelo: {nombre_modelo}")

# ==========================
# GUARDAR MODELO
# ==========================
os.makedirs("src/model", exist_ok=True)

joblib.dump((model, vectorizer), "src/model/model.pkl")

print("Modelo guardado correctamente 🚀")