import streamlit as st
import plotly.graph_objects as go

# 1. Configuración Visual (Siempre al principio)
st.set_page_config(page_title="Spam Detector AI", page_icon="🛡️")

st.title("🛡️ Analizador de Correos")
st.write("Copia y pega el mensaje para verificar su autenticidad.")

# 2. Entrada de datos
email_text = st.text_area("Contenido del correo:", height=150, placeholder="Escribe aquí...")

# 3. LÓGICA DINÁMICA (Dentro del botón)
if st.button("Verificar ahora"):
    if email_text.strip():
        # --- PROCESAMIENTO SIMULADO ---
        texto_limpio = email_text.lower()
        palabras_peligrosas = ["premio", "gratis", "ganaste", "urgente", "banco", "clic", "oferta", "beneficio"]
        
        # Conteo simple para decidir la probabilidad
        coincidencias = [p for p in palabras_peligrosas if p in texto_limpio]
        
        if len(coincidencias) > 0:
            probabilidad_spam = 80 + (len(coincidencias) * 5) # Sube según cuántas palabras detecte
            if probabilidad_spam > 100: probabilidad_spam = 100
        else:
            probabilidad_spam = 10 # Si no hay palabras raras, es bajo
            
        # --- MOSTRAR RESULTADOS VISUALES ---
        col1, col2 = st.columns(2) # Dividimos en 2 columnas para que se vea Pro
        
        with col1:
            if probabilidad_spam > 50:
                st.error("### 🚨 Resultado: SPAM")
                st.write(f"Se detectaron palabras sospechosas como: {', '.join(coincidencias)}")
            else:
                st.success("### ✅ Resultado: SEGURO")
                st.write("No se encontraron patrones evidentes de spam.")
        
        with col2:
            # Gráfica de indicador (El Paso 4)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probabilidad_spam,
                title = {'text': "Nivel de Riesgo"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if probabilidad_spam > 50 else "green"},
                    'steps': [{'range': [0, 100], 'color': "#eeeeee"}]
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.warning("⚠️ Por favor, ingresa texto para analizar.")

# 4. Pie de página o información extra (Fuera del botón)
st.divider()
st.caption("Proyecto de Machine Learning - Licenciatura en Informática")