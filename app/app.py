from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Cargar modelo
with open("../src/model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Cargar vectorizador
with open("../src/model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    palabras = []

    if request.method == "POST":
        texto = request.form["mensaje"]

        vec = vectorizer.transform([texto])
        pred = model.predict(vec)[0]

        # Resultado
        if pred == 1:
            resultado = "SPAM"
        else:
            resultado = "NO SPAM"

        # Explicación
        feature_names = vectorizer.get_feature_names_out()
        vector = vec.toarray()[0]

        for i in range(len(vector)):
            if vector[i] > 0:
                palabras.append(f"Palabra detectada: {feature_names[i]}")

    return render_template("index.html", resultado=resultado, palabras=palabras[:10])

if __name__ == "__main__":
    app.run(debug=True)