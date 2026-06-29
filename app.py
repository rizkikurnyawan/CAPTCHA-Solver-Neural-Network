# =====================================================
# OCR PROCESS (Sudah dioptimalkan untuk CAPTCHA)
# =====================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1_img, col2_res = st.columns([1, 1])

    with col1_img:
        st.subheader("Uploaded CAPTCHA")
        st.image(image, use_container_width=True)

    with st.spinner("Breaking CAPTCHA..."):
        try:
            # 1. Konversi ke OpenCV format (BGR/Grayscale)
            img_np = np.array(image)
            if len(img_np.shape) == 3:
                import cv2
                # Jika format RGBA, ubah ke RGB
                if img_np.shape[2] == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np

            # 2. CROP TOMBOL HIJAU (Mengambil 75% area kiri saja)
            # Karena CAPTCHA Anda selalu di kiri dan tombol hijau di kanan
            h, w = gray.shape
            crop_w = int(w * 0.75)
            captcha_crop = gray[:, :crop_w]

            # 3. PREPROCESSING (Thresholding untuk memperjelas teks & mengurangi garis)
            # Menggunakan Otsu's thresholding
            _, thresh = cv2.threshold(captcha_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Balikkan kembali warnanya agar teks hitam di atas background putih (disukai OCR)
            processed_img = cv2.bitwise_not(thresh)

            # Tampilkan gambar hasil pemrosesan di Streamlit untuk debugging
            # st.image(processed_img, caption="Processed Image (Input to NN)", width=150)

            # 4. FEED TO NEURAL NETWORK (EasyOCR)
            # Gunakan parameter allowlist jika CAPTCHA hanya berisi huruf kecil/angka
            result = reader.readtext(
                processed_img, 
                detail=0, 
                allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            
            # Gabungkan dan bersihkan spasi yang tidak sengaja terbaca
            extracted_text = "".join(result).replace(" ", "")

            with col2_res:
                st.subheader("Prediction Result")

                st.success(f"**Predicted Text:** {extracted_text}")

                st.text_area(
                    "Raw Output",
                    value=extracted_text,
                    height=100
                )

            if not extracted_text:
                st.warning("Neural Network gagal mengenali karakter. Coba refresh CAPTCHA baru.")
                
        except Exception as e:
            with col2_res:
                st.subheader("Error")
                st.error(f"⚠️ Terjadi kesalahan teknis: {str(e)}")
