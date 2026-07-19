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
Halaman Dashborad (Upload Data dan Tampilan Dataset)
<img width="1280" height="585" alt="image" src="https://github.com/user-attachments/assets/ab44cb47-3f60-485f-9779-16c5ed233242" />

2. **Halaman Training dan Visualisasi Model**
- Melakukan training data
- Menampilkan evaluasi matriks (MAPE, MAE, RMSE, R2 Squared)
- Menampilkan visualisasi perbandingan data real dan data prediksi
<img width="1280" height="583" alt="image" src="https://github.com/user-attachments/assets/09445899-2c91-463e-b69b-7fb266fa6c50" />

3. **Halaman Prediksi**
- Mengisi periode kasus yang ingin diprediksi (1 - 12 bulan)
- Mengisi tahun dan bulan yang ingin di prediksi
- Mengisi lag feature (jumlah kasus 6 bulan terakhir)
<img width="1280" height="433" alt="image" src="https://github.com/user-attachments/assets/b0350680-cf12-45aa-ab77-8b9f10502bb1" />

- Contoh Hasil Prediksi
<img width="1280" height="586" alt="image" src="https://github.com/user-attachments/assets/0eec87ba-d5e2-4ef9-9263-3d9ebb9710e4" />

4. **Halaman Cetak Laporan**
<img width="1280" height="586" alt="image" src="https://github.com/user-attachments/assets/003a16cd-6f26-4fb5-af13-16687a7d0270" />

# Dataset
- **Sumber** : Dinas Kesehatan Kota Padang
- **Periode** : 8 tahun (2017 - 2024)
- **Fitur yang digunakan** : Tahun, Bulan, Angka Bebas Jentik (ABJ), Jumlah Kasus

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
- RMSE : 6.53
- MAE : 4.71
- MAPE : 10.46%
- R2 Squared : 0.12

# Author
- **Author** : RAHMI MARDIAH SAPUTRI
- **Email** : rahmimardiahsaputri@gmail.com
- **Linkedin** : https://www.linkedin.com/in/rahmi-mardiah-saputri-7b6998420
