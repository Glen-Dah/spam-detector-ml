from flask import Flask, request, render_template_string
import pickle

app = Flask(
    __name__,
    template_folder="../pagina",
    static_folder="../pagina"
)

with open("../src/model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../src/model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

historial = []

STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "a", "ante", "bajo", "con", "de", "desde", "en", "entre",
    "hacia", "hasta", "para", "por", "sin", "sobre", "tras",
    "y", "e", "ni", "o", "u", "pero", "que", "si", "como",
    "yo", "tu", "el", "ella", "me", "te", "se", "nos", "le",
    "es", "son", "era", "fue", "ser", "estar", "ha", "han",
    "no", "si", "ya", "muy", "su", "sus", "al", "del", "lo",
    "este", "esta", "mi", "mas", "menos", "tambien", "les",
    "bien", "mal", "aqui", "ahi", "alli", "asi", "hoy",
    "otro", "otra", "todo", "toda", "cada", "cual", "cuyo",
    "he", "has", "hay", "ser", "sido", "have", "the", "and",
    "for", "are", "not", "with", "this", "that", "your",
    "you", "can", "will", "from", "they", "all", "been",
    "get", "its", "our", "out", "one", "but", "had", "was",
}

# ── Categorías de palabras clave → razón específica ──────────────────────────
CATEGORIAS = [
    {
        "palabras": {"prize", "winner", "won", "award", "premio", "ganador", "ganaste",
                     "felicidades", "congratulations", "selected", "elegido", "reward"},
        "razon": "Simula notificar un premio o sorteo falso para obtener datos personales."
    },
    {
        "palabras": {"free", "gratis", "gratuito", "gift", "regalo", "giveaway",
                     "freebie", "complimentary", "sin costo", "costo cero"},
        "razon": "Ofrece productos o servicios gratuitos como gancho para captar al usuario."
    },
    {
        "palabras": {"click", "link", "enlace", "url", "visit", "visita", "open",
                     "abre", "here", "aqui", "download", "descarga", "access", "accede"},
        "razon": "Incita a hacer clic en enlaces sospechosos que pueden ser maliciosos."
    },
    {
        "palabras": {"password", "contraseña", "account", "cuenta", "verify", "verifica",
                     "confirm", "confirma", "login", "credentials", "credenciales",
                     "security", "seguridad", "update", "actualiza", "banco", "bank"},
        "razon": "Suplanta a una entidad oficial para robar credenciales o datos bancarios."
    },
    {
        "palabras": {"money", "dinero", "cash", "efectivo", "earn", "gana", "income",
                     "ingreso", "profit", "ganancia", "investment", "inversion",
                     "dollar", "dolar", "peso", "bitcoin", "crypto", "rich", "rico",
                     "millionaire", "millonario", "fortune", "fortuna"},
        "razon": "Promete ganancias económicas fáciles o inversiones fraudulentas."
    },
    {
        "palabras": {"urgent", "urgente", "immediate", "inmediato", "now", "ahora",
                     "expire", "vence", "limited", "limitado", "hurry", "apurate",
                     "deadline", "last chance", "ultima oportunidad", "today", "hoy"},
        "razon": "Usa lenguaje de urgencia falsa para presionar al usuario a actuar rápido."
    },
    {
        "palabras": {"cheap", "barato", "discount", "descuento", "offer", "oferta",
                     "sale", "promo", "deal", "rebaja", "saving", "ahorro",
                     "best price", "mejor precio", "lowest", "mas bajo"},
        "razon": "Promete descuentos o precios irreales para atraer compradores."
    },
    {
        "palabras": {"weight", "peso", "diet", "dieta", "lose", "pierde", "fat",
                     "grasa", "slim", "delgado", "pill", "pastilla", "supplement",
                     "suplemento", "cure", "cura", "miracle", "milagro", "health",
                     "salud", "viagra", "pharmacy", "farmacia", "medication", "medicamento"},
        "razon": "Publicita productos milagrosos de salud o medicamentos sin respaldo médico."
    },
    {
        "palabras": {"job", "trabajo", "hire", "contrata", "employment", "empleo",
                     "remote", "remoto", "work from home", "desde casa", "salary",
                     "salario", "apply", "aplica", "position", "puesto", "career"},
        "razon": "Ofrece empleos o ingresos remotos falsos para obtener datos personales."
    },
    {
        "palabras": {"loan", "prestamo", "credit", "credito", "debt", "deuda",
                     "approved", "aprobado", "refinance", "refinancia", "mortgage",
                     "hipoteca", "finance", "financiamiento", "interest", "interes"},
        "razon": "Ofrece préstamos o créditos fáciles con condiciones engañosas o fraudulentas."
    },
    {
        "palabras": {"porn", "sex", "adult", "adulto", "nude", "desnudo", "xxx",
                     "dating", "citas", "meet", "conoce", "singles", "solteros"},
        "razon": "Contiene contenido adulto o de citas no solicitado."
    },
    {
        "palabras": {"unsubscribe", "cancel", "remove", "eliminar", "opt-out",
                     "stop receiving", "dejar de recibir", "mailing list", "lista de correo"},
        "razon": "Usa tácticas de desuscripción engañosas típicas de listas de correo masivo."
    },
]


def generar_razon_local(mensaje, palabras_clave):
    """
    Genera una razón específica analizando el contenido del mensaje
    y las palabras clave detectadas, sin necesidad de API externa.
    """
    texto = mensaje.lower()
    palabras_lower = [p.lower() for p in palabras_clave]

    # Buscar coincidencias en cada categoría
    mejor_categoria = None
    mejor_score = 0

    for cat in CATEGORIAS:
        score = 0
        # Coincidencias en palabras clave detectadas por el modelo
        for pk in palabras_lower:
            for trigger in cat["palabras"]:
                if trigger in pk or pk in trigger:
                    score += 2
        # Coincidencias directas en el texto del mensaje
        for trigger in cat["palabras"]:
            if trigger in texto:
                score += 1
        if score > mejor_score:
            mejor_score = score
            mejor_categoria = cat

    if mejor_categoria and mejor_score > 0:
        return mejor_categoria["razon"]

    # Fallback con las palabras detectadas si no hay categoría clara
    if palabras_clave:
        muestra = ", ".join(p.lower() for p in palabras_clave[:3])
        return f"Contiene términos sospechosos asociados a spam: {muestra}."

    return "Presenta estructura y patrones lingüísticos típicos de correo no deseado."


def filtrar_palabras(palabras_raw):
    filtradas = []
    vistas = set()
    for p in palabras_raw:
        limpia = p.strip().lower()
        partes = limpia.split()
        if len(partes) > 1:
            if any(parte in STOPWORDS or len(parte) < 3 for parte in partes):
                continue
        else:
            if limpia in STOPWORDS or len(limpia) < 4:
                continue
        ya_cubierta = False
        for vista in vistas:
            if limpia in vista or vista in limpia:
                if len(limpia) <= len(vista):
                    ya_cubierta = True
                    break
                else:
                    vistas.discard(vista)
                    filtradas = [f for f in filtradas if f.lower() != vista]
        if not ya_cubierta and limpia not in vistas:
            vistas.add(limpia)
            filtradas.append(p.upper())
        if len(filtradas) >= 10:
            break
    return filtradas


def calcular_confianza(model, vec, pred):
    try:
        proba = model.predict_proba(vec)[0]
        return int(round(proba[1] * 100)) if pred == 1 else int(round(proba[0] * 100))
    except AttributeError:
        pass
    try:
        import math
        score = model.decision_function(vec)[0]
        prob = 1 / (1 + math.exp(-score))
        return int(round(prob * 100)) if pred == 1 else int(round((1 - prob) * 100))
    except AttributeError:
        pass
    return 85


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    palabras  = []
    confianza = None
    razon     = None

    if request.method == "POST":
        text = request.form["mensaje"]

        if not text.strip():
            return render_template_string(
                open("../pagina/index.html", encoding="utf-8").read(),
                resultado="Ingresa un mensaje valido",
                palabras=[], confianza=None, razon=None,
                historial=historial, spam_count=0, ham_count=0
            )

        vec  = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        confianza = calcular_confianza(model, vec, pred)

        if pred == 1:
            resultado     = "Detectado como SPAM"
            feature_names = vectorizer.get_feature_names_out()
            vector        = vec.toarray()[0]
            palabras_raw  = [feature_names[i] for i in range(len(vector)) if vector[i] > 0]
            palabras      = filtrar_palabras(palabras_raw)
            razon         = generar_razon_local(text, palabras)
        else:
            resultado = "Mensaje limpio"
            palabras  = []
            razon     = None

        historial.append({
            "mensaje":   text,
            "resultado": resultado,
            "palabras":  palabras,
            "confianza": confianza,
            "razon":     razon,
        })

    spam_count = sum(1 for item in historial if "SPAM" in item["resultado"])
    ham_count  = len(historial) - spam_count

    return render_template_string(
        open("../pagina/index.html", encoding="utf-8").read(),
        resultado=resultado,
        palabras=palabras,
        confianza=confianza,
        razon=razon,
        historial=historial,
        spam_count=spam_count,
        ham_count=ham_count
    )


if __name__ == "__main__":
    app.run(debug=True)