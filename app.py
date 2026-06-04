from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import pandas as pd
# import mysql.connector # Buka comment ini jika sudah menyambungkan database

app = Flask(__name__)

# --- 1. MEMUAT MODEL MACHINE LEARNING ---
try:
    model = joblib.load('models/random_forest_cabai.pkl')
except:
    model = None # Biar tidak error saat pertama kali dijalankan jika model belum ada

# --- 2. ROUTE FOR FRONTEND (Mengembalikan Halaman HTML) ---
@app.route('/')
@app.route('/beranda')
def home():
    return render_template('beranda.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/kalender')
def kalender():
    return render_template('kalender.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Logika validasi password admin di sini (Gunakan hashing!)
        return redirect(url_for('home'))
    return render_template('login.html')

# --- 3. ROUTE FOR BACKEND / API (Mengembalikan Data JSON) ---
@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model belum dilatih atau tidak ditemukan'}), 500
    
    # Menerima input data dari frontend (misal data fitur lag atau tanggal)
    data = request.json
    
    # Contoh format data untuk prediksi (Sesuaikan dengan fitur Random Forest Anda)
    # df = pd.DataFrame([data])
    # prediksi = model.predict(df)
    
    # Dummy hasil prediksi untuk contoh
    hasil_prediksi = [35000, 36000, 38000, 42000, 40000, 41000, 45000] 
    
    return jsonify({
        'status': 'success',
        'prediksi_7_hari': hasil_prediksi
    })

if __name__ == '__main__':
    app.run(debug=True) # debug=True membuat server otomatis restart saat kode diubah