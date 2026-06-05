import pandas as pd
import numpy as np
import os

def proses_data_cabai():
    print("--- Memulai Proses Pengolahan Data ---")
    
    # Menentukan folder tempat dataset disimpan
    folder_data = "dataset"
    
    path_harga = os.path.join(folder_data, 'Tabel Harga Berdasarkan Komoditas.xlsx')
    path_raya = os.path.join(folder_data, 'Data_Hari_Raya.xlsx')
    path_output = os.path.join(folder_data, 'data_cabai_siap_ai.csv')
    
    # 1. MEMBACA DATA EXCEL
    try:
        df_harga = pd.read_excel(path_harga)
        df_raya = pd.read_excel(path_raya)
        print("✔ Berhasil membaca file Excel dari folder dataset.")
    except Exception as e:
        print(f"❌ Error: Gagal membaca file Excel di folder '{folder_data}'.")
        print(f"Detail Error: {e}")
        return

    # 2. TRANSFORMASI DATA HARGA (Mengubah format melebar menjadi memanjang)
    try:
        # id_vars disesuaikan dengan kolom asli Anda: 'No' dan 'Komoditas (Rp)'
        df_harga_long = df_harga.melt(
            id_vars=['No', 'Komoditas (Rp)'], 
            var_name='Tanggal', 
            value_name='Harga'
        )
        
        # Membersihkan spasi acak pada teks tanggal (contoh: '03/ 01/ 2022' -> '03/01/2022')
        df_harga_long['Tanggal'] = df_harga_long['Tanggal'].astype(str).str.replace(' ', '')
        
        # Mengubah teks menjadi tipe data tanggal resmi Python (datetime)
        df_harga_long['Tanggal'] = pd.to_datetime(df_harga_long['Tanggal'], format='%d/%m/%Y', errors='coerce')
        
        # --- PERBAIKAN UTAMA: MEMBERSIHKAN FORMAT TEKS PADA HARGA ---
        # Menghapus tanda koma (,) dan spasi agar karakter seperti '86,700' menjadi '86700'
        df_harga_long['Harga'] = df_harga_long['Harga'].astype(str).str.replace(',', '').str.replace(' ', '')
        
        # Mengubah tipe data string/teks tersebut menjadi angka asli (numerik)
        df_harga_long['Harga'] = pd.to_numeric(df_harga_long['Harga'], errors='coerce')
        
        # Menghapus baris jika ada proses konversi tanggal atau harga yang gagal (NaT / NaN)
        df_harga_long = df_harga_long.dropna(subset=['Tanggal', 'Harga'])
        
        # Mengurutkan data berdasarkan runtutan tanggal (Time-Series)
        df_harga_long = df_harga_long.sort_values('Tanggal').reset_index(drop=True)
        print("✔ Berhasil mengubah struktur data harga dan mengonversinya ke format angka numerik.")
    except Exception as e:
        print(f"❌ Error saat transformasi struktur kolom: {e}")
        return

    # 3. FITUR TAMBAHAN UNTUK AI (Feature Engineering)
    # Sekarang operasi matematika ini dijamin aman karena 'Harga' sudah berupa angka
    df_harga_long['lag_1'] = df_harga_long['Harga'].shift(1)
    df_harga_long['rolling_mean_7'] = df_harga_long['Harga'].rolling(window=7).mean()

    # 4. INTEGRASI POLA HARI RAYA DI BALI
    try:
        # Deteksi nama kolom tanggal di file Data_Hari_Raya.xlsx secara otomatis
        kolom_tgl_raya = 'Tanggal' if 'Tanggal' in df_raya.columns else df_raya.columns[0]
        
        df_raya['Tanggal_Raya'] = pd.to_datetime(df_raya[kolom_tgl_raya], errors='coerce')
        df_raya = df_raya.dropna(subset=['Tanggal_Raya'])
        daftar_hari_raya = df_raya['Tanggal_Raya'].dt.date.tolist()

        # Menyiapkan kolom penanda fitur pola hari raya
        df_harga_long['fase_persiapan'] = 0
        df_harga_long['fase_puncak'] = 0
        df_harga_long['fase_pasca'] = 0

        # Menentukan status fase berdasarkan jarak hari ke hari raya
        for idx, row in df_harga_long.iterrows():
            tgl_sekarang = row['Tanggal'].date()
            
            for tgl_raya in daftar_hari_raya:
                selisih_hari = (tgl_sekarang - tgl_raya).days
                
                if -7 <= selisih_hari <= -1:
                    df_harga_long.at[idx, 'fase_persiapan'] = 1
                elif selisih_hari == 0:
                    df_harga_long.at[idx, 'fase_puncak'] = 1
                elif 1 <= selisih_hari <= 3:
                    df_harga_long.at[idx, 'fase_pasca'] = 1
                    
        print("✔ Berhasil memetakan fitur fase Hari Raya.")
    except Exception as e:
        print(f"❌ Error saat sinkronisasi pola Hari Raya: {e}")
        return

    # 5. MENYIMPAN DATASET BERSIH YANG SIAP DIKONSUMSI AI
    # Menghapus baris kosong akibat proses pembuatan fitur rolling_mean/lag di awal-awal baris
    df_bersih = df_harga_long.dropna().reset_index(drop=True)
    
    # Menyimpan file ke dalam folder dataset
    df_bersih.to_csv(path_output, index=False)
    print(f"\n🎉 Keren! File siap latih sukses dibuat di: {path_output}")

if __name__ == "__main__":
    proses_data_cabai()