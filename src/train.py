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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

# ==========================
# DESCARGAR RECURSOS
# ==========================
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')

# ==========================
# LIMPIEZA
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
data = pd.read_csv("data/spam.csv", encoding='latin-1')

data = data.rename(columns={
    'label': 'etiqueta',
    'message': 'message'
})


data = data.dropna()

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
# BALANCEAR DATOS 🔥
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

# ==========================
# TF-IDF PRO
# ==========================
vectorizador = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.9
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
# MODELO FINAL
# ==========================
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)

print("\n=== MODELO FINAL (NAIVE BAYES) ===")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# ==========================
# GUARDAR MODELO
# ==========================
joblib.dump((modelo, vectorizador), "app/model.pkl")

print("\nModelo guardado correctamente 🚀")