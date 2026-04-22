from flask import Flask, request, render_template_string
import pickle
import os

# Flask usará la carpeta "pagina"
app = Flask(
    __name__,
    template_folder="../pagina",
    static_folder="../pagina"
)

# Cargar modelo
with open("../src/model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Cargar vectorizador
with open("../src/model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Historial global
historial = []



@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    palabras = []

    if request.method == "POST":
        text = request.form["mensaje"]

        # Validación
        if not text.strip():
            return render_template_string(
                open("../pagina/index.html", encoding="utf-8").read(),
                resultado="⚠️ Ingresa un mensaje válido",
                palabras=[],
                historial=historial,
                spam_count=0,
                ham_count=0
            )

        # Vectorizar
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        # Resultado
        if pred == 1:
            resultado = "🚨 SPAM detectado"
        else:
            resultado = "✅ Mensaje limpio"

        # Explicación de palabras detectadas
        feature_names = vectorizer.get_feature_names_out()
        vector = vec.toarray()[0]

        for i in range(len(vector)):
            if vector[i] > 0:
                palabras.append(feature_names[i])

        # Historial
        historial.append({
            "mensaje": text,
            "resultado": resultado
        })

    spam_count = sum(1 for item in historial if "SPAM" in item["resultado"])
    ham_count = len(historial) - spam_count

    return render_template_string(
        open("../pagina/index.html", encoding="utf-8").read(),
        resultado=resultado,
        palabras=palabras[:10],
        historial=historial,
        spam_count=spam_count,
        ham_count=ham_count
    )

if __name__ == "__main__":
    app.run(debug=True)
    