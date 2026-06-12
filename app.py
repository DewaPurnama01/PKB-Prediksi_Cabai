import os, sys, json, shutil, io as io_module
import pandas as pd
import numpy as np
import joblib
from functools import wraps
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, send_file)

app = Flask(__name__)
app.secret_key = "kelompok4_cabai_bali_super_secret_key_2026"

BASE_DIR  = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

FITUR_COLS = FITUR_NUMERIK = FITUR_BINER = None
model_rf = model_lr = scaler = None

def load_models():
    global model_rf, model_lr, scaler, FITUR_COLS, FITUR_NUMERIK, FITUR_BINER
    try:
        model_rf      = joblib.load(os.path.join(MODEL_DIR, 'model_random_forest.pkl'))
        model_lr      = joblib.load(os.path.join(MODEL_DIR, 'model_linear_regression.pkl'))
        scaler        = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        FITUR_COLS    = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
        FITUR_NUMERIK = joblib.load(os.path.join(MODEL_DIR, 'numeric_columns.pkl'))
        bin_path      = os.path.join(MODEL_DIR, 'binary_columns.pkl')
        FITUR_BINER   = joblib.load(bin_path) if os.path.exists(bin_path) else                         [c for c in FITUR_COLS if c not in FITUR_NUMERIK]
        return True
    except Exception as e:
        print(f"[WARN] Gagal muat model: {e}")
        return False

load_models()

CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'data_cabai_siap_ai.csv')
HR_PATH  = os.path.join(BASE_DIR, 'dataset', 'Data_Hari_Raya.xlsx')
LOG_PATH = os.path.join(BASE_DIR, 'dataset', 'upload_log.json')

# ─────────────────── Decorator ───────────────────

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─────────────────── Helper Functions ───────────────────

def get_hari_raya_dates():
    try:
        df = pd.read_excel(HR_PATH)
        df['tanggal'] = pd.to_datetime(df['Tanggal'])
        return df
    except:
        return pd.DataFrame(columns=['tanggal', 'Nama_Hari_Raya'])

def get_fase(tanggal, hari_raya_dates):
    diffs = [(tanggal - hr).days for hr in hari_raya_dates]
    masa_depan = [-d for d in diffs if d <= 0]
    if not masa_depan:
        return 'normal', 999, 999
    d = min(masa_depan)
    masa_lalu  = [x for x in diffs if x > 0]
    hari_sejak = min(masa_lalu) if masa_lalu else 999
    if   d <= 2:  fase = 'puncak'
    elif d <= 5:  fase = 'persiapan'
    elif d <= 10: fase = 'pasca'
    else:         fase = 'normal'
    return fase, d, hari_sejak

def build_row(next_date, history, hr_dates):
    # history berisi harga HISTORIS (t-1, t-2, dst.) — TIDAK ada harga next_date.
    # Fitur yang dibuat di sini harus IDENTIK dengan fitur di preprocess_data.py.
    h = history['harga'].values
    n = len(h)
    if n < 30:
        return None

    l1  = float(h[-1])                    # lag_1  = harga kemarin (t-1)
    l2  = float(h[-2])  if n >= 2  else l1 # lag_2  = harga 2 hari lalu (t-2)
    l3  = float(h[-3])  if n >= 3  else l1 # lag_3
    l7  = float(h[-7])  if n >= 7  else l1 # lag_7
    l14 = float(h[-14]) if n >= 14 else l1 # lag_14
    l21 = float(h[-21]) if n >= 21 else l1 # lag_21

    # Rolling stats: gunakan h[-N:] yang berisi h[-N]..h[-1]
    # Ini setara dengan harga.shift(1).rolling(N).mean() di preprocessing
    r7  = float(np.mean(h[-7:]))
    r14 = float(np.mean(h[-14:]))
    r30 = float(np.mean(h[-30:]))
    # ddof=1 agar konsisten dengan pandas .rolling().std() default
    s7  = float(np.std(h[-7:],  ddof=1)) if len(h[-7:])  > 1 else 0.0
    s14 = float(np.std(h[-14:], ddof=1)) if len(h[-14:]) > 1 else 0.0

    fase, hari_ke, hari_sejak = get_fase(next_date, hr_dates)

    return {
        'hari_dalam_seminggu': next_date.dayofweek,
        'bulan':               next_date.month,
        'kuartal':             next_date.quarter,
        'hari_dalam_bulan':    next_date.day,
        'lag_1': l1, 'lag_2': l2, 'lag_3': l3,
        'lag_7': l7, 'lag_14': l14, 'lag_21': l21,
        'rolling_mean_7': r7, 'rolling_mean_14': r14, 'rolling_mean_30': r30,
        'rolling_std_7':  s7, 'rolling_std_14':  s14,
        # selisih = perubahan harga kemarin dibanding hari sebelumnya
        'selisih_1hari':   l1 - l2,
        'selisih_7hari':   l1 - l7,
        'rasio_vs_mean30': l1 / r30 if r30 > 0 else 1.0,
        'hari_ke_hr':    hari_ke,
        'hari_sejak_hr': hari_sejak,
        'fase_normal':    1 if fase == 'normal'    else 0,
        'fase_pasca':     1 if fase == 'pasca'     else 0,
        'fase_persiapan': 1 if fase == 'persiapan' else 0,
        'fase_puncak':    1 if fase == 'puncak'    else 0,
    }

def generate_forecast_7_days(model_type='random_forest'):
    if model_rf is None:
        load_models()
    model = model_rf if model_type == 'random_forest' else model_lr

    df_csv = pd.read_csv(CSV_PATH)
    df_csv['tanggal'] = pd.to_datetime(df_csv['tanggal'])
    df_csv = df_csv.sort_values('tanggal').reset_index(drop=True)
    history   = df_csv[['tanggal', 'harga']].tail(35).copy().reset_index(drop=True)
    last_date = history['tanggal'].max()
    hr_df     = get_hari_raya_dates()
    hr_dates  = hr_df['tanggal'].tolist()

    # ── Batas prediksi untuk mencegah divergensi LR multi-step ──────────────
    # Linear Regression rentan diverge saat lag prediksi diumpankan kembali ke model.
    # Solusi: clip setiap langkah ke ±30% dari harga dasar (terakhir aktual).
    base_price  = float(history['harga'].iloc[-1])
    p_hist_min  = float(df_csv['harga'].min()) * 0.50
    p_hist_max  = float(df_csv['harga'].max()) * 1.80

    list_prediksi = []
    for i in range(1, 8):
        next_date = last_date + pd.Timedelta(days=i)
        row_data  = build_row(next_date, history, hr_dates)
        if row_data is None:
            list_prediksi.append({'tgl': next_date.strftime('%Y-%m-%d'),
                                  'harga': base_price, 'fase': 'normal'})
            continue

        # Buat DataFrame dan scale fitur numerik
        X_row = pd.DataFrame([row_data])
        num_np = scaler.transform(X_row[FITUR_NUMERIK].values.astype(float))
        num_df = pd.DataFrame(num_np, columns=FITUR_NUMERIK)
        bin_df = X_row[[c for c in FITUR_COLS if c not in FITUR_NUMERIK]].reset_index(drop=True)
        X_scaled = pd.concat([num_df, bin_df], axis=1)[FITUR_COLS]

        pred_val = float(model.predict(X_scaled)[0])

        # Clip 1: ±30% dari harga dasar (mencegah divergensi autoregresif)
        pred_val = float(np.clip(pred_val, base_price * 0.70, base_price * 1.30))
        # Clip 2: batas absolut dari rentang historis
        pred_val = float(np.clip(pred_val, p_hist_min, p_hist_max))

        fase, _, _ = get_fase(next_date, hr_dates)
        list_prediksi.append({'tgl': next_date.strftime('%Y-%m-%d'),
                              'harga': pred_val, 'fase': fase})
        history = pd.concat([history,
                              pd.DataFrame({'tanggal': [next_date], 'harga': [pred_val]})],
                             ignore_index=True)

    return df_csv.tail(10), list_prediksi

def get_metrics():
    path = os.path.join(MODEL_DIR, 'metrics.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def get_list_hari_raya():
    try:
        df = pd.read_excel(HR_PATH)
        df.columns = df.columns.str.strip()
        df['Tanggal'] = pd.to_datetime(df['Tanggal']).dt.strftime('%Y-%m-%d')
        return df.to_dict(orient='records')
    except:
        return []

def _read_hr_df():
    """Baca Excel hari raya, tambah kolom opsional jika belum ada."""
    df = pd.read_excel(HR_PATH)
    df.columns = df.columns.str.strip()
    if 'Kategori_Fase' not in df.columns:
        df['Kategori_Fase'] = 'H-1 (Puncak Permintaan)'
    if 'Catatan' not in df.columns:
        df['Catatan'] = ''
    return df

def _save_hr_df(df):
    df.to_excel(HR_PATH, index=False)

def _read_upload_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return []

def _write_upload_log(entry):
    history = _read_upload_log()
    history.insert(0, entry)
    with open(LOG_PATH, 'w') as f:
        json.dump(history[:20], f)

def _ekstrak_harga_bali_dari_excel(path_excel):
    """Baca Excel wide-format dan kembalikan DataFrame tanggal+harga Bali saja."""
    df_harga = pd.read_excel(path_excel)
    df_bali  = df_harga[df_harga['Komoditas (Rp)'].astype(str).str.strip() == 'Bali'].copy()
    if df_bali.empty:
        df_bali = df_harga.iloc[1:2].copy()
    id_cols  = ['No', 'Komoditas (Rp)']
    df_long  = df_bali.melt(id_vars=id_cols, var_name='Tgl_str', value_name='Harga')
    df_long['Tgl_str'] = df_long['Tgl_str'].astype(str).str.replace(' ', '')
    df_long['tanggal'] = pd.to_datetime(df_long['Tgl_str'], format='%d/%m/%Y', errors='coerce')
    df_long['harga']   = pd.to_numeric(
        df_long['Harga'].astype(str).str.replace(',', '').str.replace(' ', ''),
        errors='coerce')
    df_out = df_long.dropna(subset=['tanggal', 'harga'])[['tanggal', 'harga']]
    return df_out[df_out['harga'] > 0].copy()


def _gabungkan_ke_akumulasi(df_baru):
    """
    Gabungkan data baru ke dalam file akumulasi historis.
    Data baru menimpa data lama untuk tanggal yang sama (data terbaru = lebih akurat).
    Kembalikan DataFrame gabungan yang sudah diurutkan.
    """
    akum_path = os.path.join(BASE_DIR, 'dataset', 'Data_Harga_Bali_Akumulasi.csv')
    frames    = []

    # Baca akumulasi yang sudah ada
    if os.path.exists(akum_path):
        df_lama = pd.read_csv(akum_path)
        df_lama['tanggal'] = pd.to_datetime(df_lama['tanggal'])
        frames.append(df_lama)

    frames.append(df_baru)

    df_gabung = (pd.concat(frames, ignore_index=True)
                   .sort_values('tanggal')
                   .drop_duplicates(subset='tanggal', keep='last')  # data baru menang
                   .reset_index(drop=True))
    df_gabung.to_csv(akum_path, index=False)
    return df_gabung


# ─────────────────── User Routes ───────────────────

@app.route('/')
def beranda():
    model_type = session.get('model_type', 'random_forest')
    try:
        _, pred_7 = generate_forecast_7_days(model_type)
        df_csv = pd.read_csv(CSV_PATH)
        harga_hari_ini = float(df_csv.sort_values('tanggal').iloc[-1]['harga'])
        prediksi_besok = pred_7[0]['harga']
    except:
        harga_hari_ini, prediksi_besok = 47188, 49591
    return render_template('beranda.html', harga_hari_ini=harga_hari_ini,
                           prediksi_besok=prediksi_besok, model_type=model_type)

@app.route('/dashboard')
def dashboard():
    model_type = request.args.get('model', session.get('model_type', 'random_forest'))
    session['model_type'] = model_type
    try:
        hist_10, pred_7 = generate_forecast_7_days(model_type)
        harga_hari_ini  = float(hist_10.iloc[-1]['harga'])
        prediksi_besok  = pred_7[0]['harga']
        label_grafik    = [d.strftime('%Y-%m-%d') for d in hist_10['tanggal']] + [d['tgl'] for d in pred_7]
        aktual_grafik   = [float(h) for h in hist_10['harga']] + [None]*len(pred_7)
        prediksi_grafik = [None]*(len(hist_10)-1) + [float(hist_10.iloc[-1]['harga'])] + [d['harga'] for d in pred_7]
    except:
        harga_hari_ini, prediksi_besok = 47188, 49591
        pred_7 = []; label_grafik = aktual_grafik = prediksi_grafik = []
    return render_template('dashboard.html',
        harga_hari_ini=harga_hari_ini, prediksi_besok=prediksi_besok,
        pred_7=pred_7, label_grafik=label_grafik,
        aktual_grafik=aktual_grafik, prediksi_grafik=prediksi_grafik,
        is_admin=False, model_type=model_type, metrics=get_metrics())

@app.route('/kalender')
def kalender():
    hr_df = get_hari_raya_dates()
    today = pd.Timestamp.today()
    mendatang, lewat = [], []
    for _, row in hr_df.sort_values('tanggal').iterrows():
        nama  = str(row['Nama_Hari_Raya']).lower()
        entry = {'tanggal': row['tanggal'].strftime('%Y-%m-%d'),
                 'nama': row['Nama_Hari_Raya'],
                 'hari_lagi': (row['tanggal'] - today).days}
        if 'galungan' in nama or 'nyepi' in nama:
            entry['level'] = 'tinggi'; entry['est_kenaikan'] = '40–60%'
        elif 'kuningan' in nama:
            entry['level'] = 'sedang'; entry['est_kenaikan'] = '18–25%'
        else:
            entry['level'] = 'rendah'; entry['est_kenaikan'] = '5–15%'
        (mendatang if entry['hari_lagi'] >= 0 else lewat).append(entry)
    lewat.sort(key=lambda x: x['hari_lagi'], reverse=True)
    return render_template('kalender.html', mendatang=mendatang, lewat=lewat)

# ─────────────────── Public API ───────────────────

@app.route('/api/forecast')
def api_forecast():
    model_type = request.args.get('model', 'random_forest')
    try:
        _, pred_7 = generate_forecast_7_days(model_type)
        df_csv = pd.read_csv(CSV_PATH)
        harga_hari_ini = float(df_csv.sort_values('tanggal').iloc[-1]['harga'])
        return jsonify({'status': 'success', 'model_type': model_type,
                        'harga_hari_ini': harga_hari_ini,
                        'prediksi_besok': pred_7[0]['harga'], 'prediksi_7hari': pred_7})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})

@app.route('/api/cek_harga')
def api_cek_harga():
    tgl_input  = request.args.get('tanggal', '')
    model_type = request.args.get('model', session.get('model_type', 'random_forest'))
    if not tgl_input:
        return jsonify({'status': 'error', 'msg': 'Tanggal kosong'})
    try:
        df_csv = pd.read_csv(CSV_PATH)
        df_csv['tanggal'] = pd.to_datetime(df_csv['tanggal']).dt.strftime('%Y-%m-%d')
        cocok = df_csv[df_csv['tanggal'] == tgl_input]
        if not cocok.empty:
            return jsonify({'status': 'success', 'tipe': 'Data Aktual',
                            'harga': float(cocok.iloc[0]['harga'])})
        _, pred_7 = generate_forecast_7_days(model_type)
        for p in pred_7:
            if p['tgl'] == tgl_input:
                return jsonify({'status': 'success', 'tipe': 'Data Prediksi AI', 'harga': p['harga']})
        return jsonify({'status': 'not_found', 'tipe': 'Di luar jangkauan', 'harga': None})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})

@app.route('/api/hari_raya')
def api_hari_raya():
    return jsonify(get_list_hari_raya())

@app.route('/api/metrics')
def api_metrics():
    return jsonify(get_metrics())

# ─────────────────── Auth ───────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        if u == 'admin' and p == 'admin123':
            session['logged_in'] = True
            return redirect(url_for('admin_dashboard'))   # ← ke halaman admin
        return render_template('login.html', error='Username atau password salah.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('beranda'))

# ─────────────────── Admin Pages ───────────────────

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    metrics = get_metrics()
    try:
        df_csv     = pd.read_csv(CSV_PATH)
        total_data = len(df_csv)
        tgl_awal   = pd.to_datetime(df_csv['tanggal']).min().strftime('%d %b %Y')
        tgl_akhir  = pd.to_datetime(df_csv['tanggal']).max().strftime('%d %b %Y')
        integritas = round((1 - df_csv['harga'].isna().mean()) * 100, 1)
    except:
        total_data = 0; tgl_awal = tgl_akhir = '-'; integritas = 0

    hr_list   = get_list_hari_raya()
    today     = pd.Timestamp.today()
    dampak_hr = []
    for item in hr_list:
        try:
            tgl = pd.to_datetime(item['Tanggal'])
            hari_lagi = (tgl - today).days
            if -5 <= hari_lagi <= 30:
                dampak_hr.append({
                    'nama': item['Nama_Hari_Raya'], 'tanggal': item['Tanggal'],
                    'hari_lagi': hari_lagi,
                    'prediksi': '+15–25%' if 'kuningan' in item['Nama_Hari_Raya'].lower()
                               else '+40–60%'
                })
        except:
            pass
    dampak_hr = sorted(dampak_hr, key=lambda x: abs(x['hari_lagi']))[:3]

    return render_template('admin_dashboard.html',
        metrics=metrics, total_data=total_data,
        tgl_awal=tgl_awal, tgl_akhir=tgl_akhir,
        integritas=integritas, dampak_hr=dampak_hr)

@app.route('/admin/hari_raya')
@admin_required
def admin_hari_raya():
    try:
        df = _read_hr_df()
        records = []
        for i, row in df.iterrows():
            records.append({
                'id': i,
                'tanggal': pd.to_datetime(row['Tanggal']).strftime('%Y-%m-%d'),
                'nama': row['Nama_Hari_Raya'],
                'kategori': row.get('Kategori_Fase', 'H-1 (Puncak Permintaan)'),
                'catatan': row.get('Catatan', '')
            })
    except:
        records = []
    return render_template('admin_hari_raya.html', records=records)

@app.route('/admin/unggah_dataset')
@admin_required
def admin_unggah_dataset():
    return render_template('admin_unggah.html', history=_read_upload_log())

# ─────────────────── Admin API ───────────────────

@app.route('/admin/latih', methods=['POST'])
@admin_required
def admin_latih():
    try:
        old_stdout = sys.stdout
        sys.stdout = buf = io_module.StringIO()
        from train_model import latih_model
        metrics = latih_model()
        sys.stdout = old_stdout
        log_output = buf.getvalue()
        load_models()
        return jsonify({'status': 'success', 'metrics': metrics, 'log': log_output})
    except Exception as e:
        sys.stdout = old_stdout if 'old_stdout' in locals() else sys.stdout
        return jsonify({'status': 'error', 'msg': str(e)})

@app.route('/admin/api/hari_raya/tambah', methods=['POST'])
@admin_required
def api_hr_tambah():
    try:
        data     = request.get_json()
        nama     = data.get('nama', '').strip()
        tanggal  = data.get('tanggal', '').strip()
        kategori = data.get('kategori', 'H-1 (Puncak Permintaan)')
        catatan  = data.get('catatan', '')
        if not nama or not tanggal:
            return jsonify({'status': 'error', 'msg': 'Nama dan tanggal wajib diisi.'})
        df = _read_hr_df()
        new_row = pd.DataFrame([{'Tanggal': pd.to_datetime(tanggal),
                                  'Nama_Hari_Raya': nama,
                                  'Kategori_Fase': kategori,
                                  'Catatan': catatan}])
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values('Tanggal').reset_index(drop=True)
        _save_hr_df(df)
        return jsonify({'status': 'success', 'msg': 'Event berhasil ditambahkan.'})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})

@app.route('/admin/api/hari_raya/edit/<int:idx>', methods=['POST'])
@admin_required
def api_hr_edit(idx):
    try:
        data = request.get_json()
        df   = _read_hr_df()
        if idx < 0 or idx >= len(df):
            return jsonify({'status': 'error', 'msg': 'Data tidak ditemukan.'})
        df.at[idx, 'Tanggal']       = pd.to_datetime(data.get('tanggal'))
        df.at[idx, 'Nama_Hari_Raya'] = data.get('nama')
        df.at[idx, 'Kategori_Fase']  = data.get('kategori')
        df.at[idx, 'Catatan']        = data.get('catatan', '')
        _save_hr_df(df)
        return jsonify({'status': 'success', 'msg': 'Event berhasil diperbarui.'})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})

@app.route('/admin/api/hari_raya/hapus/<int:idx>', methods=['POST'])
@admin_required
def api_hr_hapus(idx):
    try:
        df = _read_hr_df()
        if idx < 0 or idx >= len(df):
            return jsonify({'status': 'error', 'msg': 'Data tidak ditemukan.'})
        df = df.drop(idx).reset_index(drop=True)
        _save_hr_df(df)
        return jsonify({'status': 'success', 'msg': 'Event berhasil dihapus.'})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})

@app.route('/admin/api/preview_upload', methods=['POST'])
@admin_required
def api_preview_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'msg': 'Tidak ada file yang dikirim.'})
        f = request.files['file']
        if f.filename == '':
            return jsonify({'status': 'error', 'msg': 'Nama file kosong.'})
        fname = secure_filename(f.filename)
        ext   = fname.rsplit('.', 1)[-1].lower() if '.' in fname else ''
        if ext not in ['csv', 'xlsx', 'xls']:
            return jsonify({'status': 'error', 'msg': 'Format harus CSV atau Excel (.xlsx/.xls).'})

        # Baca seluruh file (diperlukan untuk validasi) tapi BATASI preview
        df = pd.read_csv(f) if ext == 'csv' else pd.read_excel(f)

        # Simpan temp file untuk proses selanjutnya
        temp_path = os.path.join(BASE_DIR, 'dataset', '_temp_upload.' + ext)
        f.seek(0); f.save(temp_path)

        all_cols = list(df.columns)
        total    = len(df)

        # Deteksi format wide (Excel tanggal sebagai kolom)
        non_date_cols = ['No', 'Komoditas (Rp)']
        date_cols     = [c for c in all_cols if c not in non_date_cols]
        is_wide_format = len(date_cols) > 10

        # BATASI preview ke max 8 kolom — mencegah JSON raksasa (1046 kolom = crash!)
        preview_cols = all_cols[:8]
        preview = (df[preview_cols].head(10)
                     .fillna('N/A').astype(str)
                     .to_dict(orient='records'))

        valid = 'Komoditas (Rp)' in all_cols

        # Ringkasan informatif untuk user
        if is_wide_format:
            summary = (f"{total} baris × {len(all_cols)} kolom "
                       f"(wide format: {len(date_cols)} kolom tanggal)")
        else:
            summary = f"{total} baris × {len(all_cols)} kolom"

        return jsonify({
            'status':       'success',
            'columns':      preview_cols,       # 8 kolom pertama untuk tabel preview
            'total_cols':   len(all_cols),       # total kolom asli (informasi)
            'preview':      preview,
            'total':        total,
            'ext':          ext,
            'valid':        valid,
            'filename':     fname,
            'is_wide':      is_wide_format,
            'summary':      summary
        })
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})

@app.route('/admin/api/proses_upload', methods=['POST'])
@admin_required
def api_proses_upload():
    try:
        data      = request.get_json()
        ext       = data.get('ext', 'xlsx')
        filename  = data.get('filename', 'dataset')
        temp_path = os.path.join(BASE_DIR, 'dataset', '_temp_upload.' + ext)
        if not os.path.exists(temp_path):
            return jsonify({'status': 'error', 'msg': 'File sementara tidak ditemukan. Upload ulang.'})
        # ── Perilaku APPEND: gabungkan data baru dengan data historis ──────────
        # BUKAN replace — data yang sudah ada TIDAK hilang.
        # Upload Jan-Jun 2026 di atas data 2022-2026 → hasil: 2022-Jun 2026
        if ext == 'xlsx':
            df_baru = _ekstrak_harga_bali_dari_excel(temp_path)
        else:
            df_raw  = pd.read_csv(temp_path)
            # Coba deteksi apakah CSV sudah punya kolom tanggal+harga
            if 'tanggal' in df_raw.columns and 'harga' in df_raw.columns:
                df_raw['tanggal'] = pd.to_datetime(df_raw['tanggal'])
                df_baru = df_raw[['tanggal', 'harga']].dropna()
            else:
                # Jika CSV adalah format wide, simpan dulu sebagai xlsx lalu ekstrak
                tmp_xlsx = temp_path.replace('.csv', '_conv.xlsx')
                df_raw.to_excel(tmp_xlsx, index=False)
                df_baru = _ekstrak_harga_bali_dari_excel(tmp_xlsx)
                os.remove(tmp_xlsx)

        if df_baru.empty:
            return jsonify({'status': 'error',
                            'msg': 'Tidak ada data harga Bali yang ditemukan di file ini.'})

        # Gabungkan ke akumulasi historis — Excel asli TIDAK ditimpa
        df_gabung = _gabungkan_ke_akumulasi(df_baru)

        from preprocess_data import proses_data_cabai
        proses_data_cabai()
        os.remove(temp_path)

        # Otomatis latih ulang model setelah dataset diperbarui
        try:
            import sys as _sys, io as _io
            _old_stdout = _sys.stdout
            _sys.stdout = _buf = _io.StringIO()
            from train_model import latih_model
            metrics = latih_model()
            _sys.stdout = _old_stdout
            load_models()  # Muat ulang model baru ke memori
            log_msg = _buf.getvalue()
            status_msg = (f'Dataset diperbarui & model dilatih ulang. '
                          f'RF MAE={metrics["random_forest"]["mae"]:,.0f} | '
                          f'R²={metrics["random_forest"]["r2"]}')
        except Exception as train_err:
            _sys.stdout = _old_stdout if '_old_stdout' in dir() else _sys.stdout
            status_msg = f'Dataset disimpan. Pelatihan model gagal: {train_err}'
            metrics = {}

        # Info total data akumulasi
        akum_path = os.path.join(BASE_DIR, 'dataset', 'Data_Harga_Bali_Akumulasi.csv')
        if os.path.exists(akum_path):
            df_akum = pd.read_csv(akum_path)
            df_akum['tanggal'] = pd.to_datetime(df_akum['tanggal'])
            info_akum = (f" | Total data akumulasi: {len(df_akum):,} baris "
                         f"({df_akum['tanggal'].min().strftime('%d %b %Y')} – "
                         f"{df_akum['tanggal'].max().strftime('%d %b %Y')})")
        else:
            info_akum = ""

        _write_upload_log({
            'waktu':     datetime.now().strftime('%d %b %Y, %H:%M'),
            'nama_file': filename,
            'status':    'Berhasil',
            'baris_baru': len(df_baru),
            'total_akum': len(df_gabung)
        })
        return jsonify({'status': 'success',
                        'msg': status_msg + info_akum,
                        'metrics': metrics,
                        'info_akumulasi': {
                            'total_baris': len(df_gabung),
                            'tgl_awal': str(df_gabung['tanggal'].min().date()),
                            'tgl_akhir': str(df_gabung['tanggal'].max().date()),
                            'baris_baru': len(df_baru)
                        }})
    except Exception as e:
        _write_upload_log({'waktu': datetime.now().strftime('%d %b %Y, %H:%M'),
                           'nama_file': data.get('filename', '?'), 'status': f'Gagal: {e}'})
        return jsonify({'status': 'error', 'msg': str(e)})

@app.route('/admin/template_download')
@admin_required
def admin_template_download():
    df = pd.DataFrame({
        'No':             ['I', 'II', '1', '2', '3'],
        'Komoditas (Rp)': ['Semua Provinsi', 'Bali', 'Kota Denpasar', 'Pasar Badung', 'Pasar Kumbasari'],
        '03/01/2025':     [85000, 70000, 75000, 72000, 68000],
        '04/01/2025':     [84000, 69000, 74000, 71000, 67500],
        '05/01/2025':     [83500, 68500, 73500, 70500, 67000],
    })
    buf = io_module.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name='Template_Dataset_Harga.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
