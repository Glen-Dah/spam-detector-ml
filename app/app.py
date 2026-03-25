from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

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

        

        # 🔴 VALIDACIÓN (PASO 2)
        if not text.strip():
            return render_template(
                "index.html",
                resultado="⚠️ Ingresa un mensaje válido",
                palabras=[],
                historial=historial
            )

        # 🔤 Vectorizar
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        # 📊 Resultado
        if pred == 1:
            resultado = "🚨 SPAM detectado"
        else:
            resultado = "✅ Mensaje limpio"

        # 🧠 Explicación (palabras detectadas)
        feature_names = vectorizer.get_feature_names_out()
        vector = vec.toarray()[0]

        for i in range(len(vector)):
            if vector[i] > 0:
                palabras.append(feature_names[i])

        # 🧾 HISTORIAL (PASO 3)
        historial.append({
            "mensaje": text,
            "resultado": resultado
        })

    return render_template(
        "index.html",
        resultado=resultado,
        palabras=palabras[:10],
        historial=historial
    )
    

if __name__ == "__main__":
    app.run(debug=True)