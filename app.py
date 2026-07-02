import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# ... (Konfigurasi Halaman dan Load Model tetap sama) ...

def preprocess_captcha(image):
    # 1. Konversi ke grayscale
    gray = np.array(image.convert("L"))
    
    # 2. Thresholding (Binarization) - Memisahkan teks hitam dari background putih
    # Menggunakan Otsu's thresholding agar garis pengganggu tipis hilang
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Noise Removal (Menghilangkan garis-garis tipis pengganggu)
    # Gunakan morphological opening (Erosi lalu Dilasi) untuk memutus garis tipis
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 4. Segmentasi Karakter (Opsional, tapi sangat disarankan)
    # Mencari kontur setiap karakter agar model tidak kebingungan dengan huruf yang bersambung
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sortir kontur dari kiri ke kanan
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes.sort(key=lambda b: b[0]) # Urutkan berdasarkan x (kiri ke kanan)
    
    return bounding_boxes, cleaned

# --- DI BAGIAN UI STREAMLIT ---

if uploaded_file is not None and session is not None:
    img = Image.open(uploaded_file)
    
    # Terapkan Preprocessing di atas
    boxes, processed_img = preprocess_captcha(img)
    
    predicted_text = ""
    
    # Loop untuk setiap karakter yang berhasil dipisahkan
    for x, y, w, h in boxes:
        # Crop karakter individual
        char_roi = processed_img[y:y+h, x:x+w]
        
        # Resize ke ukuran yang diharapkan model (misal 32x32)
        # Jangan resize paksa ke 200x50 untuk seluruh gambar, tapi resize per karakter!
        char_resized = cv2.resize(char_roi, (32, 32)) 
        
        # Normalisasi & ONNX Format
        img_array = char_resized.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1) # (32, 32, 1)
        img_input = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 1)
        
        # Prediksi per karakter
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: img_input})
        
        # Ambil indeks karakter dengan probabilitas tertinggi
        char_index = np.argmax(predictions[0][0])
        predicted_text += characters[char_index]
        
    # Tampilkan Hasil
    st.image(processed_img, caption="Gambar Setelah Cleanup", use_container_width=True)
    st.success(f"Hasil Prediksi Model: {predicted_text}")
