import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(page_title="Hadi Engine NN", layout="wide")

# Karakter map yang digunakan saat training
characters = "abcdefghijklmnopqrstuvwxyz0123456789"

# Muat Model ONNX secara Cache (Super Ringan & Hemat RAM)
@st.cache_resource
def load_onnx_model():
    # Pastikan file 'model.onnx' sudah di-upload ke repositori GitHub Anda
    return ort.InferenceSession("model.onnx")

try:
    session = load_onnx_model()
except Exception as e:
    st.error("Gagal memuat file model. Pastikan 'model.onnx' ada di repositori.")
    session = None

# UI Streamlit
st.title("Neural Network Image Reader (ONNX)")

uploaded_file = st.file_uploader("Unggah Gambar", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and session is not None:
    img = Image.open(uploaded_file)
    
    # Pre-processing (Samakan persis dengan setelan saat Anda training model)
    img_gray = img.convert("L")
    img_resized = img_gray.resize((200, 50))  # Sesuaikan ukuran input model Anda (Lebar, Tinggi)
    
    img_array = np.array(img_resized).astype(np.float32) / 255.0  # Wajib float32 untuk ONNX
    img_array = np.expand_dims(img_array, axis=-1)  # Menjadi shape (50, 200, 1)
    img_input = np.expand_dims(img_array, axis=0)   # Menjadi shape (1, 50, 200, 1)
    
    # Prediksi menggunakan ONNX Runtime
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: img_input})
    
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
