import re
import unicodedata
import os
import math
import joblib

from flask import Flask, request, render_template_string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

app = Flask(
    __name__,
    template_folder="../pagina",
    static_folder="../pagina"
)

# ==========================
# CARGAR MODELO
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(
    BASE_DIR,
    "src",
    "model",
    "model.pkl"
)

model, vectorizer = joblib.load(model_path)

# ==========================
# VARIABLES
# ==========================
historial = []

stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')

STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "a", "ante", "bajo", "con", "de", "desde", "en", "entre",
    "hacia", "hasta", "para", "por", "sin", "sobre", "tras",
    "y", "e", "ni", "o", "u", "pero", "que", "si", "como",
    "yo", "tu", "ella", "me", "te", "se", "nos", "le",
    "es", "son", "era", "fue", "ser", "estar", "ha", "han",
    "no", "ya", "muy", "su", "sus", "al", "del", "lo",
    "este", "esta", "mi", "mas", "menos", "tambien", "les",
    "bien", "mal", "aqui", "ahi", "alli", "asi", "hoy",
}

# ==========================
# LIMPIEZA
# ==========================
def limpiar_texto(texto):

    texto = str(texto).lower()

    texto = unicodedata.normalize(
        'NFKD',
        texto
    ).encode(
        'ascii',
        'ignore'
    ).decode(
        'utf-8'
    )

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
# CATEGORÍAS
# ==========================
CATEGORIAS = [

    {
        "palabras": {
            "premio", "ganaste", "winner",
            "prize", "reward", "felicidades"
        },

        "razon":
        "El mensaje intenta convencer al usuario de que ganó un premio o recompensa falsa para obtener información personal."
    },

    {
        "palabras": {
            "gratis", "free", "gift",
            "regalo", "beneficio"
        },

        "razon":
        "El mensaje ofrece productos o servicios gratuitos de forma sospechosa para atraer la atención del usuario."
    },

    {
        "palabras": {
            "click", "clic", "link",
            "enlace", "download",
            "descarga", "visit"
        },

        "razon":
        "El mensaje solicita ingresar a enlaces potencialmente peligrosos o maliciosos."
    },

    {
        "palabras": {
            "banco", "bank", "password",
            "cuenta", "login",
            "credenciales", "verify"
        },

        "razon":
        "El mensaje intenta obtener credenciales o información bancaria simulando ser una entidad oficial."
    },

    {
        "palabras": {
            "urgente", "urgent", "ahora",
            "immediate", "limited",
            "vence", "today"
        },

        "razon":
        "El mensaje utiliza lenguaje de urgencia para presionar al usuario a actuar rápidamente."
    },

    {
        "palabras": {
            "dinero", "money", "bitcoin",
            "crypto", "investment",
            "profit", "ganancia"
        },

        "razon":
        "El mensaje promete ganancias económicas rápidas o inversiones sospechosas."
    },

    {
        "palabras": {
            "trabajo", "empleo",
            "salary", "remote",
            "work from home"
        },

        "razon":
        "El mensaje ofrece oportunidades laborales poco realistas o sospechosas."
    },
]

# ==========================
# FILTRAR PALABRAS
# ==========================
def filtrar_palabras(palabras_raw):

    filtradas = []
    vistas = set()

    for p in palabras_raw:

        limpia = p.strip().lower()

        if limpia in STOPWORDS:
            continue

        if len(limpia) < 4:
            continue

        ya_cubierta = False

        for vista in list(vistas):

            if limpia in vista or vista in limpia:

                if len(limpia) <= len(vista):
                    ya_cubierta = True
                    break

                else:
                    vistas.discard(vista)

                    filtradas = [
                        f for f in filtradas
                        if f.lower() != vista
                    ]

        if not ya_cubierta:

            vistas.add(limpia)
            filtradas.append(p.upper())

        if len(filtradas) >= 10:
            break

    return filtradas

# ==========================
# GENERAR RAZÓN
# ==========================
def generar_razon_local(mensaje, palabras_clave):

    texto = mensaje.lower()

    palabras_lower = [
        p.lower()
        for p in palabras_clave
    ]

    mejor_categoria = None
    mejor_score = 0

    for cat in CATEGORIAS:

        score = 0

        # palabras detectadas
        for pk in palabras_lower:

            for trigger in cat["palabras"]:

                if trigger in pk or pk in trigger:
                    score += 2

        # texto completo
        for trigger in cat["palabras"]:

            if trigger in texto:
                score += 1

        if score > mejor_score:

            mejor_score = score
            mejor_categoria = cat

    if mejor_categoria and mejor_score > 0:

        return mejor_categoria["razon"]

    if palabras_clave:

        muestra = ", ".join(
            p.lower()
            for p in palabras_clave[:3]
        )

        return (
            f"El sistema detectó términos "
            f"sospechosos relacionados con spam "
            f"como: {muestra}."
        )

    return (
        "El mensaje contiene patrones "
        "lingüísticos asociados a correo no deseado."
    )

# ==========================
# CONFIANZA
# ==========================
def calcular_confianza(model, vec, pred):

    try:

        proba = model.predict_proba(vec)[0]

        if pred == 1:
            return int(round(proba[1] * 100))
        else:
            return int(round(proba[0] * 100))

    except AttributeError:

        score = model.decision_function(vec)[0]

        prob = 1 / (1 + math.exp(-score))

        if pred == 1:
            return int(round(prob * 100))
        else:
            return int(round((1 - prob) * 100))

# ==========================
# APP
# ==========================
@app.route("/", methods=["GET", "POST"])
def index():

    resultado = None
    palabras = []
    confianza = None
    razon = None

    if request.method == "POST":

        text = request.form["mensaje"]

        if not text.strip():

            return render_template_string(
                open(
                    "../pagina/index.html",
                    encoding="utf-8"
                ).read(),

                resultado="Ingresa un mensaje válido",
                palabras=[],
                confianza=None,
                razon=None,
                historial=historial,
                spam_count=0,
                ham_count=0
            )

        # ==========================
        # LIMPIAR TEXTO
        # ==========================
        texto_limpio = limpiar_texto(text)

        # ==========================
        # VECTORIZAR
        # ==========================
        vec = vectorizer.transform(
            [texto_limpio]
        )

        # ==========================
        # PROBABILIDAD
        # ==========================
        proba = model.predict_proba(vec)[0][1]

        # threshold mejorado
        if proba >= 0.60:
            pred = 1
        else:
            pred = 0

        confianza = int(round(proba * 100))

        # ==========================
        # RESULTADO
        # ==========================
        if pred == 1:

            resultado = "🚨 Detectado como SPAM"
        if pred == 1:

            resultado = "🚨 Detectado como SPAM"

            # Extraer palabras reales del texto original
            palabras_raw = re.findall(
                r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑ]+\b',
                text
            )

            palabras = filtrar_palabras(
                palabras_raw
            )

            razon = generar_razon_local(
                text,
                palabras
            )       

        else:

            resultado = "✅ Mensaje limpio"

            palabras = []

            razon = (
                "El mensaje no presenta "
                "características comunes de spam."
            )

        # ==========================
        # HISTORIAL
        # ==========================
        historial.append({

            "mensaje": text,
            "resultado": resultado,
            "palabras": palabras,
            "confianza": confianza,
            "razon": razon
        })

    spam_count = sum(

        1 for item in historial

        if "SPAM" in item["resultado"]
    )

    ham_count = len(historial) - spam_count

    return render_template_string(

        open(
            "../pagina/index.html",
            encoding="utf-8"
        ).read(),

        resultado=resultado,
        palabras=palabras,
        confianza=confianza,
        razon=razon,
        historial=historial,
        spam_count=spam_count,
        ham_count=ham_count
    )

# ==========================
# RUN
# ==========================
if __name__ == "__main__":

    app.run(debug=True)