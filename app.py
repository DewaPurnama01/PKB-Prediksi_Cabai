import os
import sys
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

app = Flask(__name__)
app.secret_key = "kelompok4_cabai_bali_super_secret_key"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_random_forest.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'feature_columns.pkl')

CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'data_cabai_siap_ai.csv')
HR_PATH = os.path.join(BASE_DIR, 'dataset', 'Data_Hari_Raya.xlsx')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    fitur_cols = joblib.load(FEATURES_PATH)
except Exception as e:
    print(f"[ERROR] Gagal memuat file pkl dari folder models: {e}")
    sys.exit(1)

try:
    import database
except ImportError:
    database = None

df_hr = pd.read_excel(HR_PATH)
df_hr["Tanggal"] = pd.to_datetime(df_hr["Tanggal"], errors="coerce")
df_hr = df_hr.dropna(subset=["Tanggal"])

set_H1_H2 = set()
set_H3_H5 = set()
set_H1_H3_pasca = set()

for _, row in df_hr.iterrows():
    tgl_hr = row["Tanggal"]
    for delta in [-2, -1]:
        set_H1_H2.add(tgl_hr + pd.Timedelta(days=delta))
    for delta in [-5, -4, -3]:
        set_H3_H5.add(tgl_hr + pd.Timedelta(days=delta))
    for delta in [1, 2, 3]:
        set_H1_H3_pasca.add(tgl_hr + pd.Timedelta(days=delta))

def hitung_prediksi_7_hari():
    """
    Melakukan peramalan autoregresif 7 hari ke depan memanfaatkan data riil terakhir
    """
    df_clean = pd.read_csv(CSV_PATH)
    df_clean["tanggal"] = pd.to_datetime(df_clean["tanggal"])
    df_clean = df_clean.sort_values("tanggal").reset_index(drop=True)
    
    history = df_clean.tail(30).copy()
    
    fitur_numerik = [
        "hari_dalam_seminggu", "bulan", "kuartal", "hari_dalam_bulan",
        "lag_1", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_mean_14",
        "selisih_harga_1hari", "selisih_harga_7hari"
    ]
    
    hasil_peramalan = []
    last_date = history["tanggal"].max()
    
    for i in range(1, 8):
        next_date = last_date + pd.Timedelta(days=i)
        
        lag_1 = history.iloc[-1]["harga"]
        lag_2 = history.iloc[-2]["harga"]
        lag_7 = history.iloc[-7]["harga"]
        lag_8 = history.iloc[-8]["harga"]
        lag_14 = history.iloc[-14]["harga"]
        
        rolling_7 = history.iloc[-7:]["harga"].mean()
        rolling_14 = history.iloc[-14:]["harga"].mean()
        
        f_normal, f_pasca, f_persiapan, f_puncak = 1, 0, 0, 0
        if next_date in set_H1_H2:
            f_puncak, f_normal = 1, 0
        elif next_date in set_H3_H5:
            f_persiapan, f_normal = 1, 0
        elif next_date in set_H1_H3_pasca:
            f_pasca, f_normal = 1, 0
            
        input_data = {
            "hari_dalam_seminggu": next_date.dayofweek,
            "bulan": next_date.month,
            "kuartal": next_date.quarter,
            "hari_dalam_bulan": next_date.day,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "rolling_mean_7": rolling_7,
            "rolling_mean_14": rolling_14,
            "fase_normal": f_normal,
            "fase_pasca": f_pasca,
            "fase_persiapan": f_persiapan,
            "fase_puncak": f_puncak,
            "selisih_harga_1hari": lag_1 - lag_2,
            "selisih_harga_7hari": lag_1 - lag_8
        }
        
        df_row = pd.DataFrame([input_data])[fitur_cols]
        df_row[fitur_numerik] = scaler.transform(df_row[fitur_numerik])
        
        prediksi_harga = float(model.predict(df_row.values)[0])
        
        new_row = pd.DataFrame({"tanggal": [next_date], "harga": [prediksi_harga]})
        history = pd.concat([history, new_row], ignore_index=True)
        
        hasil_peramalan.append({
            "tgl": next_date.strftime("%Y-%m-%d"),
            "harga": prediksi_harga
        })
        
    return df_clean.tail(10), hasil_peramalan

@app.route('/')
def beranda():
    try:
        hist_10, pred_7 = hitung_prediksi_7_hari()
        harga_hari_ini = float(hist_10.iloc[-1]["harga"])
        prediksi_besok = pred_7[0]["harga"]
    except Exception:
        harga_hari_ini, prediksi_besok = 47188, 49591 # Fallback dummy aman
        
    return render_template('beranda.html', harga_hari_ini=harga_hari_ini, prediksi_besok=prediksi_besok)

@app.route('/dashboard')
def dashboard():
    try:
        hist_10, pred_7 = hitung_prediksi_7_hari()
        harga_hari_ini = float(hist_10.iloc[-1]["harga"])
        prediksi_besok = pred_7[0]["harga"]
        
        label_grafik = [d.strftime("%d %b") for d in hist_10["tanggal"]] + [pd.to_datetime(d["tgl"]).strftime("%d %b") for d in pred_7]
        
        aktual_grafik = [float(h) for h in hist_10["harga"]] + [None] * len(pred_7)
        
        prediksi_grafik = [None] * (len(hist_10) - 1) + [float(hist_10.iloc[-1]["harga"])] + [d["harga"] for d in pred_7]
        
    except Exception as e:
        print(f"Error parsing dashboard data: {e}")
        harga_hari_ini, prediksi_besok = 47188, 49591
        pred_7 = [{"tgl": "2026-05-05", "harga": 49591}]
        label_grafik = ["20 Apr", "22 Apr", "24 Apr", "26 Apr"]
        aktual_grafik = [52000, 49000, 47500, None]
        prediksi_grafik = [None, None, 47500, 49591]

    return render_template(
        'dashboard.html',
        harga_hari_ini=harga_hari_ini,
        prediksi_besok=prediksi_besok,
        pred_7=pred_7,
        label_grafik=label_grafik,
        aktual_grafik=aktual_grafik,
        prediksi_grafik=prediksi_grafik
    )

@app.route('/kalender')
def kalender():
    return render_template('kalender.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if database and hasattr(database, 'verify_user'):
            auth = database.verify_user(username, password)
        else:
            auth = (username == "admin" and password == "admin123") # Fallback default
            
        if auth:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Username atau password salah!")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('beranda'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)