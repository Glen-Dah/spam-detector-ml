import pickle
import re

# limpiar texto igual que entrenamiento
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# cargar modelo
with open("src/model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("src/model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# input
text = input("Escribe un mensaje: ")
text_limpio = limpiar_texto(text)

# vectorizar
vec = vectorizer.transform([text_limpio])

# predecir
pred = model.predict(vec)[0]

# resultado
if pred == 1:
    print("🚨 SPAM")
else:
    print("✅ NO SPAM")

# explicación
feature_names = vectorizer.get_feature_names_out()
vector = vec.toarray()[0]

palabras = []

for i in range(len(vector)):
    if vector[i] > 0:
        palabras.append(feature_names[i])

print("\nPalabras clave detectadas:")
print(palabras[:10])