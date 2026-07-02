import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# Karakter map
characters = "abcdefghijklmnopqrstuvwxyz0123456789"
num_classes = len(characters) + 1

st.set_page_config(page_title="CRNN CAPTCHA Solver", layout="wide")
st.title("CRNN CAPTCHA Solver")

@st.cache_resource
def load_onnx_model():
    try:
        return ort.InferenceSession("model.onnx")
    except Exception as e:
        st.error(f"Gagal memuat model.onnx: {e}")
        return None

session = load_onnx_model()

uploaded_file = st.file_uploader("Unggah Gambar CAPTCHA", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if session is None:
        st.stop()

    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Gambar Input", use_container_width=True)

    # Preprocessing
    img_gray = img.convert("L")
    img_np = np.array(img_gray)
    _, img_thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_res = cv2.resize(img_thresh, (200, 50))
    
    img_array = img_res.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_input = np.expand_dims(img_array, axis=0)

    # Prediksi
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: img_input})[0]

    # --- PERBAIKAN DECODER ---
    decoded_indices = np.argmax(logits[0], axis=1)
    
    result_text = ""
    prev_idx = -1
    blank_idx = len(characters) # Indeks ke-36
    
    for idx in decoded_indices:
        if idx != blank_idx:
            if idx != prev_idx:
                result_text += characters[idx]
            prev_idx = idx
        else:
            prev_idx = -1 # Reset saat bertemu blank
            
    with col2:
        if len(result_text) == 0:
            st.error("❌ Gagal mendeteksi karakter. Model mungkin belum trained dengan baik.")
            # Tampilkan logits sebagai debug
            st.write("Logits mentah (Time step 0-5):")
            st.write(logits[0][:5])
        else:
            st.success(f"🔍 Hasil Prediksi: **{result_text}**")
