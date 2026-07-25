# dengue-prediction-rf
Peramalan jumlah kasus DBD bulanan menggunakan algoritma Random Forest di Python berdasarkan data tahun, bulan, Angka Bebas Jentik (ABJ), dan riwayat jumlah kasus.

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
1. **Halaman Login**

<img width="1280" height="586" alt="image" src="https://github.com/user-attachments/assets/7bf87936-c89b-4fe2-b70f-c45f5c86a02d" />

2. **Halaman Dashborad (Upload Data dan Tampilan Dataset)**
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



3. **Halaman Training dan Visualisasi Model**
- Melakukan training data
- Menampilkan evaluasi matriks (MAPE, MAE, RMSE, R2 Squared)
- Menampilkan visualisasi perbandingan antara data hasil prediksi dengan data aktual dari dataset
- Menampilkan selisih antara data prediksi dan data aktual
<img width="1280" height="575" alt="image" src="https://github.com/user-attachments/assets/13203ec5-ecd3-4a56-909a-67f257150f47" />


4. **Halaman Prediksi**
- Mengisi periode kasus yang ingin diprediksi (1 - 12 bulan)
- Mengisi tahun dan bulan yang ingin di prediksi
- Mengisi lag feature (jumlah kasus 6 bulan terakhir)

<img width="1280" height="380" alt="image" src="https://github.com/user-attachments/assets/d65db3fb-8b54-4a46-9200-5e384ae61425" />
- Contoh Hasil Prediksi
<img width="1280" height="447" alt="image" src="https://github.com/user-attachments/assets/270e9fc5-7907-4488-973d-7ce14cb2756f" />


5. **Halaman Cetak Laporan**
<img width="1280" height="584" alt="image" src="https://github.com/user-attachments/assets/521cdf98-4502-4d11-9ff3-28bdc6ebb547" />


# Dataset
- **Sumber                 :** Dinas Kesehatan Kota Padang
- **Periode                :** 8 tahun (2017 - 2024)
- **Fitur yang digunakan   :** Tahun, Bulan, Angka Bebas Jentik (ABJ), Jumlah Kasus

# Teknologi yang digunakan
- **Framework :** Flask

- **Algoritma :** Random Forest Regression

- **Backend :** Python
  
- **Frontend :** HTML, CSS

# Metodologi yang Digunakan
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
   * Menerapkan metodologi gabungan antara **Grid Search CV** dan **TimeSeriesSplit** (dengan 3 *fold*) untuk proses *Hyperparameter Tuning*. Langkah ini memastikan model dievaluasi secara ketat berdasarkan urutan kronologis waktu tanpa risiko kebocoran data masa depan.
   * Parameter model yang dioptimasi meliputi jumlah pohon (`n_estimators`), kedalaman maksimum (`max_depth`), jumlah fitur maksimal (`max_features`), jumlah sampel minimum untuk *split* (`min_samples_split`), dan jumlah sampel minimum pada *leaf* (`min_samples_leaf`).
6. **Evaluasi Model:** Mengukur akurasi prediksi menggunakan metrik evaluasi standar regresi, yaitu RMSE, MAE, MAPE, dan $R^2$ Squared.

# Hasil dan Evaluasi Model
- **RMSE       :** 6.53
- **MAE        :** 4.71
- **MAPE       :** 10.46%
- **R2 Squared :** 0.12

# 📁 Struktur Direktori Proyek

```text
dengue-prediction-rf/
├── static/                  
│   ├── files                # Menyimpan dataset yang digunakan
│   │   └── 260114_dbd_padangkotor.csv
│   ├── Images               # Menyimpan gambar
│   │   ├── logo.png
│   │   └── nyamuk.jpg        
│   └── style.css            # Mengatur tampilan halaman
├── model/                   # Tempat menyimpan file model RF yang sudah dilatih
│   └── rf_model.joblib
├── data/                    # Tempat untuk menyimpan data
│   ├── data_cleaned.csv     # Tempat untuk menyimpan data yang sudah diproses 
│   └── hasil_prediksi.csv   # Tempat untuk menyimpan data hasil prediksi
├── templates/               # Tempat untuk menyimpan script tampilan web
│   ├── base.html     
│   ├── dashboard.html       # Script tampilan dashboard
│   ├── laporan.html         # Script tampilan laporan
│   ├── login.html           # Script tampilan login
│   ├── prediksi.html        # Script tampilan prediksi 
│   └── visualisasi.html     # Script tampilan visualisasi
├── app.py                   # File utama untuk menjalankan website/dashboard
├── train.py                 # Script untuk training data
├── train_model.ipynb        # Script untuk training data dengan format .ipynb
├── requirements.txt         # Daftar library Python (pandas, scikit-learn, dll)
└── README.md                # Dokumentasi proyek
```

# Author
- **Author   :** RAHMI MARDIAH SAPUTRI
- **Email    :** rahmimardiahsaputri@gmail.com
- **Linkedin :** https://www.linkedin.com/in/rahmi-mardiah-saputri-7b6998420
