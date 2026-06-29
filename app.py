import streamlit as st
import numpy as np
import cv2

import onnxruntime as ort

@st.cache_resource
def load_model():
    return ort.InferenceSession("model.onnx")
    
from PIL import Image
import tensorflow as tf

# Konfigurasi Halaman
st.set_page_config(page_title="Hadi Engine NN", layout="wide")

# Karakter map yang digunakan saat training (contoh: huruf kecil + angka)
characters = "abcdefghijklmnopqrstuvwxyz0123456789"

# Muat Model Neural Network secara Cache
@st.cache_resource
def load_custom_model():
    # Pastikan file 'model.keras' sudah di-upload ke repositori Anda
    return tf.keras.models.load_model("model.keras")

try:
    model = load_custom_model()
except Exception as e:
    st.error("Gagal memuat file model. Pastikan 'model.keras' ada di repositori.")
    model = None

# UI Streamlit
st.title("Neural Network Image Reader")

uploaded_file = st.file_uploader("Unggah Gambar", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file)
    
    # Pre-processing agar sesuai dengan input model (contoh: Grayscale & Resize)
    img_gray = img.convert("L")
    img_resized = img_gray.resize((200, 50))  # Sesuaikan ukuran input model Anda
    
    img_array = np.array(img_resized) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=-1)  # Tambah channel dimension (50, 200, 1)
    img_input = np.expand_dims(img_array, axis=0)   # Tambah batch dimension (1, 50, 200, 1)
    
    # Prediksi menggunakan Neural Network
    predictions = model.predict(img_input)
    
    # Menerjemahkan output matriks menjadi teks string
    predicted_text = ""
    for pred in predictions:
        char_index = np.argmax(pred[0])
        predicted_text += characters[char_index]
        
    # Tampilkan Hasil
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Gambar Input", use_container_width=True)
    with col2:
        st.success(f"Hasil Prediksi Model: {predicted_text}")
