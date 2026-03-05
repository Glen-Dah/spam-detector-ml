import pickle

# Cargar modelo
with open("src/model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Cargar vectorizador
with open("src/model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Pedir mensaje
text = input("Escribe un mensaje: ")

# Vectorizar
text_vec = vectorizer.transform([text])

# Predecir
pred = model.predict(text_vec)[0]

if pred == 1:
    print("SPAM")
else:
    print("NO SPAM")