# 📌 Sentimen myPertamina — Streamlit App

Aplikasi ini adalah alat analisis sentimen berbasis **Streamlit** yang memanfaatkan **scikit-learn** dan **TF-IDF Vectorizer** untuk mengklasifikasi ulasan terkait layanan myPertamina menjadi kategori sentimen (positif, negatif, atau multikelas).

## 🚀 Fitur Utama
- **🏠 Dashboard**: Menampilkan ringkasan dataset, distribusi label sentimen, histogram panjang teks, dan contoh ulasan.
- **📂 Upload Data**: Mengunggah file CSV dengan pilihan kolom teks dan label secara fleksibel.
- **🧠 Latih Model**: Mendukung pelatihan model sentimen biner (positif/negatif) atau multikelas.
- **⚡ Prediksi Real-Time**: Memasukkan teks ulasan untuk memperoleh prediksi sentimen instan.

## 📂 Struktur Proyek
```
mypertamina-sentiment/
│
├── app.py               # Aplikasi utama Streamlit
├── myPertamina.csv      # Dataset contoh (opsional)
├── requirements.txt     # Daftar dependensi Python
└── README.md            # Dokumentasi proyek
```

## ⚙️ Instalasi Lokal
1. **Clone repository**:
```bash
git clone https://github.com/DumaAritonang18/Sentimen_mypertamina.git
cd Sentimen_mypertamina
```
2. **Buat virtual environment dan aktifkan**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. **Install dependensi**:
```bash
pip install -r requirements.txt
```
4. **Jalankan aplikasi**:
```bash
streamlit run app.py
```

## 🌐 Deploy ke Streamlit Cloud
1. Push repository ini ke **GitHub**.
2. Buka [Streamlit Cloud](https://share.streamlit.io/).
3. Hubungkan akun GitHub dan pilih repository ini.
4. Atur **Main file path** ke `app.py`.
5. Klik **Deploy** untuk memulai.

## 📦 requirements.txt
```
streamlit
pandas
numpy
scikit-learn
matplotlib
plotly
joblib
```
## Link Streamlit.io 
https://sentimenmypertamina-nxsq5oz7klpzeztmtef7hh.streamlit.app/

## 📜 Lisensi
Proyek ini dibuat untuk Tugas UTS Mata Kuliah Pemrograman Bahasa Alami - Duma Zindy Aritonang
