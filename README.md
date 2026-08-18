# 🦟 Dengue Case Forecasting Web Application using Random Forest Regression

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-RandomForest-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Aplikasi web berbasis data (*data-driven web application*) yang dirancang khusus untuk memprediksi dan meramalkan jumlah kasus bulanan Demam Berdarah Dengue (DBD). Proyek ini memanfaatkan pendekatan *Machine Learning Pipeline* dengan mengimplementasikan algoritma **Random Forest Regressor** untuk memodelkan pola non-linear pada data deret waktu (*time-series*).

Sistem prediktif ini mengintegrasikan indikator lingkungan berupa **Angka Bebas Jentik (ABJ)** serta rekayasa fitur berbasis **Lag Features (riwayat tren kasus 6 bulan terakhir)** sebagai variabel prediktor utama. Dikembangkan menggunakan backend **Flask (Python)** dan arsitektur frontend yang responsif, aplikasi ini ditujukan untuk membantu instansi kesehatan dalam melakukan deteksi dini, perencanaan logistik medis, dan pengambilan keputusan preventif yang lebih akurat.

---

## 📑 Daftar Isi
- [Fitur](#fitur)
- [Tampilan Website](#tampilan-website)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)
- [Cara Penggunaan](#cara-penggunaan)
- [Dataset](#dataset)
- [Metodologi](#metodologi-yang-digunakan)
- [Struktur Direktori Proyek](#-struktur-direktori-proyek)
- [Hasil dan Evaluasi Model](#hasil-dan-evaluasi-model)
- [Author](#author)
- [License](#license)
  
# Fitur
- Login
- Upload Dataset
- Tampilan Dataset
- Training Data
- Tampilan Visualisasi Data Training dan Data Testing
- Prediksi Data
- Cetak Hasil Prediksi
- Logout

# Tampilan Website
## 1. **Halaman Login**

<img width="1280" height="586" alt="image" src="https://github.com/user-attachments/assets/7bf87936-c89b-4fe2-b70f-c45f5c86a02d" />

## 2. **Halaman Dashborad (Upload Data dan Tampilan Dataset)**
- Menampilkan jumlah data yanng tersedia
- Kasus tertinggi
- Kasus terendah
- Rata-rata kasus per bulan
- Periode data yanng di upload
- Bar chart untuk missing value dalam dataset
- Chart untuk menampilkan Total Kasus Per Bulan
<img width="1280" height="580" alt="image" src="https://github.com/user-attachments/assets/72de1656-e64d-4a23-8ba7-196130a4cb99" />

- Trend fluktuasi kasus DBD dalam dataset
<img width="1280" height="455" alt="image" src="https://github.com/user-attachments/assets/8484f772-35b1-4e5a-80a1-b71b1a0042be" />

- Insight dataset
- tampilan dataset dalam tabel
<img width="1280" height="584" alt="image" src="https://github.com/user-attachments/assets/0bac726e-f0b9-4f4e-acaa-c532d9509622" />


## 3. **Halaman Training dan Visualisasi Model**
- Melakukan training data
- Menampilkan evaluasi matriks (MAPE, MAE, RMSE, R2 Squared)
- Menampilkan visualisasi perbandingan antara data hasil prediksi dengan data aktual dari dataset
- Menampilkan selisih antara data prediksi dan data aktual
<img width="1280" height="575" alt="image" src="https://github.com/user-attachments/assets/13203ec5-ecd3-4a56-909a-67f257150f47" />


## 4. **Halaman Prediksi**
- Mengisi periode kasus yang ingin diprediksi (1 - 12 bulan)
- Mengisi tahun dan bulan yang ingin di prediksi
- Mengisi lag feature (jumlah kasus 6 bulan terakhir)

<img width="1280" height="380" alt="image" src="https://github.com/user-attachments/assets/d65db3fb-8b54-4a46-9200-5e384ae61425" />
- Contoh Hasil Prediksi
<img width="1280" height="447" alt="image" src="https://github.com/user-attachments/assets/270e9fc5-7907-4488-973d-7ce14cb2756f" />


## 5. **Halaman Cetak Laporan**
<img width="1280" height="584" alt="image" src="https://github.com/user-attachments/assets/521cdf98-4502-4d11-9ff3-28bdc6ebb547" />

## Teknologi yang digunakan
- **Framework :** Flask

- **Algoritma :** Random Forest Regression

- **Backend :** Python
  
- **Frontend :** HTML, CSS

## Instalasi & Cara Menjalankan

### Prasyarat
- Python 3.10 atau lebih baru
- pip
- (opsional) virtualenv

### Langkah Instalasi

1. **Clone repository**
```bash
   git clone https://github.com/<username>/dengue-prediction-rf.git
   cd dengue-prediction-rf
```

2. **Buat dan aktifkan virtual environment** (disarankan)
```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Jalankan aplikasi**
```bash
   python app.py
```

5. **Buka di browser**

http://localhost:5000


> Model yang sudah dilatih tersimpan di `model/rf_model.joblib`, jadi aplikasi bisa langsung dipakai untuk prediksi tanpa training ulang. Untuk melatih ulang model dari awal, jalankan `train.py` atau buka `train_model.ipynb`.

## Cara Penggunaan

1. **Login** ke aplikasi menggunakan akun yang tersedia.
2. **Upload dataset** kasus DBD (format `.csv`) melalui halaman Dashboard.
3. Lihat **ringkasan dan visualisasi** dataset (tren kasus, missing value, insight otomatis).
4. Masuk ke halaman **Training** untuk melatih ulang model dan melihat evaluasi performa (MAPE, MAE, RMSE, R²).
5. Masuk ke halaman **Prediksi**, isi periode yang ingin diprediksi beserta lag feature (jumlah kasus 6 bulan terakhir).
6. **Cetak hasil prediksi** melalui halaman Laporan.

## Dataset
- **Sumber                 :** Dinas Kesehatan Kota Padang
- **Periode                :** 8 tahun (2017 - 2024)
- **Fitur yang digunakan   :** Tahun, Bulan, Angka Bebas Jentik (ABJ), Jumlah Kasus

## Metodologi yang Digunakan
Proyek ini menerapkan metodologi berbasis *Machine Learning Pipeline* untuk data deret waktu (*time-series*). Tahapan metodologi yang dilakukan meliputi:

1. **Pengumpulan Data:** Menggunakan data sekunder dari Dinas Kesehatan Kota Padang periode 2017 - 2024 yang mencakup variabel waktu, indikator lingkungan (ABJ), dan historis kasus.
2. **Pra-pemrosesan Data (Preprocessing):**
   * Pembersihan data (*data cleaning*) dan penanganan nilai yang hilang (*missing values*).
   * Transformasi data waktu (Tahun dan Bulan).
3. **Rekayasa Fitur (Feature Engineering):**
   * Pembuatan fitur *Lag* (menggunakan data riwayat jumlah kasus dari 6 bulan terakhir sebagai prediktor).
   * Integrasi fitur Angka Bebas Jentik (ABJ) sebagai variabel prediktor tambahan.
4. **Split Data:** Menggunakan metode *Hold Out* untuk membagi ddata menjadi *Training Data* untuk melatih model sebanyak 12 data dan *Testing Data* untuk menguji performa prediksi sebanyak 78 data.
5. **Pembangunan Model (Modeling):**
   * Menggunakan algoritma **Random Forest Regressor** di Python untuk mempelajari pola non-linear pada data kasus DBD.
   * Menerapkan metodologi gabungan antara **Grid Search CV** dan **TimeSeriesSplit** (dengan 3 *fold*) untuk proses *Hyperparameter Tuning*, langkah ini memastikan model dievaluasi secara ketat berdasarkan urutan kronologis waktu tanpa risiko kebocoran data masa depan.
   * Parameter model yang dioptimasi meliputi jumlah pohon (`n_estimators`), kedalaman maksimum (`max_depth`), jumlah fitur maksimal (`max_features`), jumlah sampel minimum untuk *split* (`min_samples_split`), dan jumlah sampel minimum pada *leaf* (`min_samples_leaf`).
6. **Evaluasi Model:** Mengukur akurasi prediksi menggunakan metrik evaluasi standar regresi, yaitu RMSE, MAE, MAPE, dan $R^2$ Squared.

# 📁 Struktur Direktori Proyek

```text
dengue-prediction-rf/
├── static/                  
│   ├── files                    # Menyimpan dataset yang digunakan
│   │   └── 260114_dbd_padangkotor.csv
│   ├── Images                   # Menyimpan gambar
│   │   ├── logo.png
│   │   └── nyamuk.jpg        
│   └── style.css                # Mengatur tampilan halaman
├── model/                       # Tempat menyimpan file model RF yang sudah dilatih
│   └── rf_model.joblib
├── data/                        # Tempat untuk menyimpan data
│   ├── data_cleaned.csv         # Tempat untuk menyimpan data yang sudah diproses 
│   └── hasil_prediksi.csv       # Tempat untuk menyimpan data hasil prediksi
├── templates/                   # Tempat untuk menyimpan script tampilan web
│   ├── base.html     
│   ├── dashboard.html          # Script tampilan dashboard
│   ├── laporan.html            # Script tampilan laporan
│   ├── login.html              # Script tampilan login
│   ├── prediksi.html           # Script tampilan prediksi 
│   └── visualisasi.html        # Script tampilan visualisasi
├── Dataset DBD 2017-22024.csv  # Dataset yang digunakan untuk melatih model
├── app.py                      # File utama untuk menjalankan website/dashboard
├── train.py                    # Script untuk training data
├── train_model.ipynb           # Script untuk training data dengan format .ipynb
├── LICENSE
├── last_file.txt            
├── training_status.txt         
└── README.md                   # Dokumentasi proyek
```

## Hasil dan Evaluasi Model
- **RMSE       :** 6.53
- **MAE        :** 4.71
- **MAPE       :** 10.46%
- **R2 Squared :** 0.12
> **Catatan:** Nilai R² yang masih rendah (0.12) mengindikasikan model belum sepenuhnya menangkap variasi kasus DBD, kemungkinan dipengaruhi oleh jumlah data training yang terbatas dan minimnya variabel eksternal (misalnya curah hujan, kepadatan penduduk, atau data iklim). Rencana pengembangan selanjutnya meliputi penambahan fitur lingkungan tambahan dan eksplorasi algoritma lain (misalnya XGBoost atau LSTM) untuk pola time-series yang lebih kompleks.

---
## Author

**Rahmi Mardiah Saputri**

[![Email](https://img.shields.io/badge/Email-rahmimardiahsaputri%40gmail.com-red?logo=gmail&logoColor=white)](mailto:rahmimardiahsaputri@gmail.com)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rahmi%20Mardiah%20Saputri-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rahmi-mardiah-saputri-7b6998420)

## License

Proyek ini dilisensikan di bawah [MIT License](LICENSE) — bebas digunakan, dimodifikasi, dan didistribusikan dengan mencantumkan atribusi.
