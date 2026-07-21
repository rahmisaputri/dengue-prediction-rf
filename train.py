import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# =============================
# PATH CONFIG
# =============================
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "files")
LAST_FILE_PATH = os.path.join(BASE_DIR, "last_file.txt")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model", "rf_model.joblib")
RESULT_SAVE_PATH = os.path.join(BASE_DIR, "data", "hasil_prediksi.csv")
CLEANED_SAVE_PATH = os.path.join(BASE_DIR, "data", "data_cleaned.csv")

# =============================
# LOAD FILE TERAKHIR
# =============================
def get_last_csv():
    if not os.path.exists(LAST_FILE_PATH):
        raise FileNotFoundError("Belum ada data yang diupload")

    with open(LAST_FILE_PATH, "r") as f:
        filename = f.read().strip()

    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        raise FileNotFoundError("File terakhir tidak ditemukan")

    return path

# =============================
# TRAINING PIPELINE
# =============================
def train_model():
    print("🔄 Training dimulai...")

    # ========= LOAD =========
    csv_path = get_last_csv()
    df = pd.read_csv(csv_path, sep=";", index_col="No")

    # ========= DROP MISSING KRITIS =========
    df = df.dropna(subset=['Jumlah Kasus', 'Tahun', 'Bulan'])

    # ========= CLEAN ABJ =========
    df['ABJ'] = (
        df['ABJ']
        .astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    df['ABJ'] = pd.to_numeric(df['ABJ'], errors='coerce')

    # isi missing ABJ → mean per bulan
    df['ABJ'] = df.groupby('Bulan')['ABJ'].transform(
        lambda x: x.fillna(x.mean())
    )

    # fallback mean global
    df['ABJ'] = df['ABJ'].fillna(df['ABJ'].mean())

    # persen → desimal
    df['ABJ'] = (df['ABJ'] / 100).round(4)

    # ========= MAP BULAN =========
    bulan_map = {
        'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'Mei':5, 'Juni':6,
        'Juli':7, 'Agu':8, 'Sep':9, 'Okt':10, 'Nov':11, 'Des':12
    }
    df['Bulan'] = df['Bulan'].map(bulan_map)

    # drop gagal mapping
    df = df.dropna(subset=['Bulan'])
    df['Bulan'] = df['Bulan'].astype(int)

    # ========= SORT TIME =========
    df = df.sort_values(by=['Tahun', 'Bulan']).reset_index(drop=True)

    # ========= LAG FEATURE =========
    for i in range(1, 7):
        df[f'Lag{i}'] = df['Jumlah Kasus'].shift(i)

    df['Diff'] = df['Lag1'] - df['Lag2']

    # hapus NaN akibat lag
    df = df.dropna().reset_index(drop=True)

    # ========= SAVE CLEANED =========
    os.makedirs(os.path.dirname(CLEANED_SAVE_PATH), exist_ok=True)
    df.to_csv(CLEANED_SAVE_PATH, index=False)

    # ========= SPLIT TIME SERIES =========
    X = df.drop(['Jumlah Kasus', 'Tahun', 'Bulan'], axis=1)
    y = df['Jumlah Kasus']

    test_size = 12
    split_index = len(X) - test_size

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # ========= MODEL =========
    tscv = TimeSeriesSplit(n_splits=3)

    param_grid = {
        'n_estimators': [200],
        'max_depth': [3, 5],
        'min_samples_split': [3, 5],
        'min_samples_leaf': [3, 5],
        'max_features': ['sqrt'],
        'bootstrap': [True]
    }

    rf = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # ========= PREDIKSI =========
    y_pred = best_model.predict(X_test)

    df_hasil = pd.DataFrame({
        'Tahun': df.iloc[split_index:]['Tahun'].values,
        'Bulan': df.iloc[split_index:]['Bulan'].values,
        'Data Aktual': y_test.values,
        'Data Prediksi': y_pred
    })

    # ========= SAVE =========
    os.makedirs(os.path.dirname(RESULT_SAVE_PATH), exist_ok=True)
    df_hasil.to_csv(RESULT_SAVE_PATH, index=False)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)

    print("✅ Training selesai & hasil tersimpan")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    train_model()
