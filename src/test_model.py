import pickle

# Cargar modelo
with open("src/model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Cargar vectorizador
with open("src/model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Mensajes para probar
tests = [
    "Win money now",
    "Free prize claim now",
    "Urgent call now",
    "Hola como estas",
    "Vamos a comer mañana",
    "Meeting tomorrow at 10"
]

print("Pruebas del modelo:\n")

for text in tests:

    vec = vectorizer.transform([text])

    pred = model.predict(vec)[0]

    if pred == 1:
        result = "SPAM"
    else:
        result = "HAM"

    print(f"{text} -> {result}")