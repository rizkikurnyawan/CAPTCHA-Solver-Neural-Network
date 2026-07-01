from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
import onnxruntime as ort
import os

app = FastAPI(title="CAPTCHA Solver API for UiPath")

# Karakter map (harus sama dengan model training)
characters = "abcdefghijklmnopqrstuvwxyz0123456789"

# Muat model ONNX
MODEL_PATH = "model.onnx"
if os.path.exists(MODEL_PATH):
    session = ort.InferenceSession(MODEL_PATH)
else:
    raise FileNotFoundError("File model.onnx tidak ditemukan!")

@app.post("/predict")
async def predict_captcha(file: UploadFile = File(...)):
    try:
        # 1. Baca file gambar yang dikirim dari UiPath
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # 2. Pre-processing (harus sama persis dengan Streamlit/Colab)
        img_resized = cv2.resize(img, (200, 50))
        img_array = img_resized.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_input = np.expand_dims(img_array, axis=0)
        
        # 3. Prediksi dengan ONNX Runtime
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: img_input})
        
        # 4. Terjemahkan output matriks menjadi string teks
        predicted_text = ""
        for pred in predictions:
            char_index = np.argmax(pred[0])
            predicted_text += characters[char_index]
            
        # 5. Kembalikan respons JSON ke UiPath
        return {"status": "success", "captcha_result": predicted_text}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Menjalankan server lokal di port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
