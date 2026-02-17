import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# ----------------------------------
# Streamlit Seitenkonfiguration
# ----------------------------------
st.set_page_config(
    page_title="KI Bildklassifikation",
    page_icon="🤖",
    layout="centered"
)

st.title("🍎🌳⚽ KI-Bildklassifikation")
st.write("Diese App unterscheidet zwischen **Apfel**, **Baum** und **Ball**.")

# ----------------------------------
# Modell laden (nur einmal laden!)
# ----------------------------------
@st.cache_resource
def load_keras_model():
    model = load_model("keras_model.h5", compile=False)
    return model

try:
    model = load_keras_model()
except Exception as e:
    st.error("❌ Modell konnte nicht geladen werden.")
    st.stop()

# ----------------------------------
# Labels laden
# ----------------------------------
def load_labels():
    try:
        with open("labels.txt", "r") as f:
            labels = f.readlines()
        labels = [label.strip().split(" ", 1)[-1] for label in labels]
        return labels
    except:
        st.error("❌ labels.txt konnte nicht geladen werden.")
        st.stop()

class_names = load_labels()

# ----------------------------------
# Bild Upload
# ----------------------------------
uploaded_file = st.file_uploader(
    "📤 Lade ein Bild hoch",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Hochgeladenes Bild", use_container_width=True)

    # ----------------------------------
    # Bild vorbereiten
    # ----------------------------------
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # ----------------------------------
    # Vorhersage
    # ----------------------------------
    with st.spinner("🔍 Klassifiziere Bild..."):
        prediction = model.predict(data)
        prediction = prediction[0]

    index = np.argmax(prediction)
    best_class = class_names[index]
    confidence_score = float(prediction[index])

    # ----------------------------------
    # Ergebnis anzeigen
    # ----------------------------------
    st.subheader("🎯 Ergebnis")
    st.success(f"**Vorhersage:** {best_class}")
    st.write(f"**Sicherheit:** {confidence_score:.2%}")

    # ----------------------------------
    # Wahrscheinlichkeitsdiagramm
    # ----------------------------------
    st.subheader("📊 Wahrscheinlichkeiten")

    df = pd.DataFrame({
        "Klasse": class_names,
        "Wahrscheinlichkeit": prediction
    }).sort_values("Wahrscheinlichkeit", ascending=False)

    st.bar_chart(df.set_index("Klasse"))

    # Detailanzeige
    for _, row in df.iterrows():
        st.write(f"{row['Klasse']}: {row['Wahrscheinlichkeit']:.2%}")
