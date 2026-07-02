import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

st.title("CAPTCHA Solver AI")

# 1. Load Model ONNX
@st.cache_resource
def load_model():
    return ort.InferenceSession("model.onnx")

session = load_model()

# 2. Upload File HARUS dilakukan sebelum pengecekan
uploaded_file = st.file_uploader("Unggah Gambar CAPTCHA", type=["png", "jpg", "jpeg"])

# 3. Baru lakukan pengecekan (Ini baris ke-34 Anda)
if uploaded_file is not None and session is not None:
    # Baca gambar
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar Input")

    # ... (Masukkan kode preprocessing gambar dan prediksi ONNX Anda di sini) ...
    # Pastikan Anda menyesuaikan preprocessing dengan model CRNN Anda (resize 200x50, dll.)
    
    st.success("Model berhasil memproses gambar!")
