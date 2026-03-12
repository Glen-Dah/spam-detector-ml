import pickle
import numpy as np

# =========================
# CARGAR MODELO
# =========================

with open("src/model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("src/model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Obtener palabras que conoce el modelo
feature_names = vectorizer.get_feature_names_out()

# =========================
# PEDIR MENSAJE
# =========================

text = input("Escribe un mensaje: ")

# =========================
# VECTORIZAR TEXTO
# =========================

text_vec = vectorizer.transform([text])

# =========================
# PREDECIR
# =========================

pred = model.predict(text_vec)[0]

# Probabilidad de la predicción
proba = model.predict_proba(text_vec)[0]
spam_prob = proba[1]

# =========================
# RESULTADO
# =========================

if pred == 1:
    print(f"\nResultado: SPAM ({spam_prob*100:.2f}% probabilidad)")
else:
    print(f"\nResultado: HAM ({(1-spam_prob)*100:.2f}% probabilidad)")

# =========================
# EXPLICACION
# =========================

indices = text_vec.nonzero()[1]
words = [feature_names[i] for i in indices]

print("\nPalabras detectadas en el mensaje:")

if len(words) == 0:
    print("No se detectaron palabras relevantes")
else:
    for w in words:
        print("-", w)