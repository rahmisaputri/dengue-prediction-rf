from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import joblib
import os
import pandas as pd
import numpy as np
import subprocess

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

USERNAME = "admin"
PASSWORD = "12345"
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

STATUS_PATH = "training_status.txt"


class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_last_file():
    try:
        with open("last_file.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def write_status(status):
    with open(STATUS_PATH, "w") as f:
        f.write(status)


def read_status():
    if not os.path.exists(STATUS_PATH):
        return "NONE"
    return open(STATUS_PATH).read().strip()


def clear_upload_folder():
    folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(folder):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == USERNAME and password == PASSWORD:
            session["login"] = True
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Username atau password salah")

    return render_template("login.html")


@app.route('/dashboard', methods=["GET", "POST"])
def dashboard():
    if "login" not in session:
        return redirect(url_for("login"))

    form = UploadFileForm()
    success = False
    show_table = False
    data = None
    columns = None

    jumlah_data = None
    rata_rata = None
    periode_tahun = None

    error = None

    last_file = get_last_file()
    df = None

    if last_file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], last_file)
        if os.path.exists(filepath) and last_file.endswith('.csv'):
            df = pd.read_csv(filepath, sep=';')
            df = df.fillna("Data tidak tersedia")

    if form.validate_on_submit():
        file = form.file.data
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            if not filename.lower().endswith('.csv'):
                error = "File tidak valid. Silakan upload file dengan format .csv"
            else:
                clear_upload_folder()
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                df = pd.read_csv(filepath, sep=';')

                df = df.rename(columns={
                    'tahun': 'Tahun',
                    'Tahun': 'Tahun',
                    'bulan': 'Bulan',
                    'Bulan': 'Bulan',
                    'jumlah kasus': 'Jumlah Kasus',
                    'Jumlah Kasus': 'Jumlah Kasus',
                    'abj': 'Angka Bebas Jentik (ABJ)',
                    'ABJ': 'Angka Bebas Jentik (ABJ)'
               })
                
                df = df.fillna("Data tidak tersedia")

                with open("last_file.txt", "w") as f:
                    f.write(filename)

                success = "Data berhasil di upload."
        else:
            error = "File tidak valid. Silakan upload file .csv"

    if df is not None:
        jumlah_data = len(df)
        rata_rata = round(df['Jumlah Kasus'].mean(), 2)
        periode_tahun = f"{df['Tahun'].min()} - {df['Tahun'].max()}"

        columns = df.columns.tolist()
        data = df.values.tolist()
        show_table = True

    return render_template(
        'dashboard.html',
        form=form,
        success=success,
        show_table=show_table,
        columns=columns,
        data=data,
        jumlah_data=jumlah_data,
        rata_rata=rata_rata,
        periode_tahun=periode_tahun,
        error=error
    )


# VISUALISASI
@app.route('/visualisasi', methods=["GET", "POST"])
def visualisasi():
    if "login" not in session:
        return redirect(url_for("login"))

    labels, aktual, prediksi = [], [], []
    mae = rmse = r2 = mape = None
    hasil_lengkap = None

    status = read_status()

    def load_hasil():
        nonlocal labels, aktual, prediksi, mae, rmse, r2, mape, hasil_lengkap

        if not os.path.exists("data/hasil_prediksi.csv"):
            raise FileNotFoundError("File hasil_prediksi.csv belum tersedia")

        hasil_lengkap = pd.read_csv("data/hasil_prediksi.csv")

        if hasil_lengkap.empty:
            raise ValueError("File hasil_prediksi.csv kosong")

        required_cols = [
            "Tahun", "Bulan",
            "Data Aktual",
            "Data Prediksi"
        ]
        for col in required_cols:
            if col not in hasil_lengkap.columns:
                raise ValueError(f"Kolom '{col}' tidak ditemukan di CSV")

        labels = [f"{int(t)}-{int(b):02d}" for t, b in zip(hasil_lengkap["Tahun"], hasil_lengkap["Bulan"])]

        aktual = pd.to_numeric(hasil_lengkap["Data Aktual"], errors="coerce").fillna(0).tolist()
        prediksi = pd.to_numeric(hasil_lengkap["Data Prediksi"], errors="coerce").fillna(0).tolist()

        mae = round(mean_absolute_error(aktual, prediksi), 2)
        rmse = round(np.sqrt(mean_squared_error(aktual, prediksi)), 2)
        r2 = round(r2_score(aktual, prediksi), 2)

        valid = np.array(aktual) != 0
        if valid.any():
            mape = round(np.mean(np.abs((np.array(aktual)[valid] - np.array(prediksi)[valid]) / np.array(aktual)[valid])) * 100, 2)

    try:
        if request.method == "POST":
            write_status("RUNNING")

            subprocess.run(["python", "train.py"], check=True)

            write_status("DONE")

            # Redirect ke GET setelah selesai training
            return redirect(url_for("visualisasi"))

        else:
            if os.path.exists("data/hasil_prediksi.csv"):
                load_hasil()

    except Exception as e:
        write_status("ERROR")
        flash(f"Visualisasi gagal: {e}", "danger")
        print("ERROR VISUALISASI:", e)

    return render_template(
        "visualisasi.html",
        labels=labels,
        aktual=aktual,
        prediksi=prediksi,
        mae=mae,
        rmse=rmse,
        r2=r2,
        mape=mape,
        hasil_lengkap=hasil_lengkap,
        status=status
    )

#PREDIKSI
@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    if "login" not in session:
        return redirect(url_for("login"))

    # Default kosong
    hasil_prediksi = session.get("hasil_prediksi", [])
    labels = session.get("labels", [])
    prediksi_chart = session.get("prediksi_chart", [])
    form_data = session.get("form_data", {})

    if request.method == 'POST':
        # ambil input
        xbulan = int(request.form['xbulan'])
        tahun = int(request.form['tahun'])
        bulan = int(request.form['bulan'])
        abj = float(request.form['abj'])
        lag1 = float(request.form['lag1'])
        lag2 = float(request.form['lag2'])
        lag3 = float(request.form['lag3'])
        lag4 = float(request.form['lag4'])
        lag5 = float(request.form['lag5'])
        lag6 = float(request.form['lag6'])

        model = joblib.load('model/rf_model.joblib')

        current_lags = [lag1, lag2, lag3, lag4, lag5, lag6]

        hasil_prediksi = []
        labels = []
        prediksi_chart = []

        for step in range(1, xbulan + 1):
            diff = current_lags[0] - current_lags[1]

            fitur = np.array([[abj,
                               current_lags[0],
                               current_lags[1],
                               current_lags[2],
                               current_lags[3],
                               current_lags[4],
                               current_lags[5],
                               diff]])

            def round_standard(x):
                return int(np.floor(x + 0.5))

            pred = model.predict(fitur)[0]
            pred_int = round_standard(pred)
            bulan_nama = [
                "Januari", "Februari", "Maret", "April","Mei",
                "Juni", "Juli", "Agustus", "September", "Oktober",
                "November", "Desember"
                ]
            hasil_prediksi.append({
                    'tahun': tahun,
                    'bulan': bulan_nama[bulan -1],
                    'prediksi': round(pred, 2),
                    'ket': pred_int
                })

            labels.append(f"{bulan_nama[bulan-1]} ({bulan:02d})")
            prediksi_chart.append(round(pred, 2))

            current_lags = [pred] + current_lags[:-1]

            bulan += 1
            if bulan > 12:
                bulan = 1
                tahun += 1

        # SIMPAN KE SESSION supaya tetap ada
        session["hasil_prediksi"] = hasil_prediksi
        session["labels"] = labels
        session["prediksi_chart"] = prediksi_chart
        session["form_data"] = request.form.to_dict()
        session.modified = True

        # redirect agar tidak duplikasi saat refresh
        return redirect(url_for("prediksi"))

    return render_template(
        'prediksi.html',
        hasil_prediksi=hasil_prediksi,
        labels=labels,
        prediksi_chart=prediksi_chart,
        form_data=form_data
    )

#CETAK LAPORAN
from datetime import datetime

@app.route('/cetak_laporan')
def cetak_laporan():
    now = datetime.now()
    nama_bulan = [
         "Januari", "Februari", "Maret", "April","Mei",
         "Juni", "Juli", "Agustus", "September", "Oktober",
         "November", "Desember"
        ]
    
    nomor_dokumen = now.strftime("DBD-PRED/%Y/%m/%d/%H%M%S")
    tanggal_laporan = f"{now.day} {nama_bulan[now.month-1]} {now.year}"
    return render_template(
        "laporan.html",
        nomor_dokumen=nomor_dokumen,
        tanggal_laporan=tanggal_laporan,
        hasil_prediksi=session.get("hasil_prediksi", []),
        print_mode=True
    )




# LOGOUT
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == '__main__':
    app.run(debug=True)
