import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect **Healthy**, **Powdery**, or **Rust** disease.")


@st.cache_resource
def load_cnn_model():
    model = load_model("plant_disease_model.h5")  
    return model

model = load_cnn_model()

class_names = ["Healthy", "Powdery", "Rust"]

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=400)

  
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

  
    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing leaf image..."):
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

        st.success(f"### ✅ Prediction: **{predicted_class}**")
        st.info(f"### 📊 Confidence: **{confidence:.2f}%**")
