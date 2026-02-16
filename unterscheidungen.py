import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bildklassifikation", layout="centered")

st.title("🌳🍎⚽ Baum, Apfel oder Ball")
st.write("Lade ein Bild hoch und erhalte eine Vorhersage mit Wahrscheinlichkeitsanzeige.")

# Modell laden (nur einmal)
@st.cache_resource
def load_model():
    try:
        from keras.models import load_model  

model = load_model('keras_model.h5', compile=False)  

        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None

model = load_model()


# Labels laden
def load_labels():
    try:
        labels = ['Baum', 'Apfel', 'Ball'] 

        return labels
    except Exception as e:
        st.error(f"Fehler beim Laden der Labels: {e}")
        return []

labels = load_labels()

# Bild vorbereiten
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array.astype(np.float32) / 127.5 - 1
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Upload-Funktion
uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    processed_image = preprocess_image(image)

    # Vorhersage mit Modell
    if model:
        prediction = model.predict(processed_image)
        probabilities = prediction[0]

        # Beste Klasse
        index = np.argmax(probabilities)
        confidence = probabilities[index]

        st.subheader("🔎 Ergebnis")
        st.success(f"Vorhersage: **{labels[index]}**")
        st.write(f"Sicherheit: **{confidence * 100:.2f}%**")

        # ---- Wahrscheinlichkeits-Balkendiagramm ----
        st.subheader("📊 Wahrscheinlichkeiten aller Klassen")

        # Wahrscheinlichkeiten und Labels sortieren
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_probabilities = probabilities[sorted_indices] * 100

        fig, ax = plt.subplots()

        colors = ["green" if i == 0 else "gray" for i in range(len(sorted_labels))]

        ax.bar(sorted_labels, sorted_probabilities, color=colors)
        ax.set_ylim([0, 100])
        ax.set_ylabel("Wahrscheinlichkeit (%)")
        ax.set_title("Modell-Vorhersage")

        st.pyplot(fig)
