import pandas as pd
import numpy as np
import os

def _baca_dan_gabung_sumber(folder_data, path_harga, path_seed):
    """
    Membaca data harga Bali dari dua sumber dan menggabungkannya:
    1. Data_Historis_Bali_Seed.csv  — data historis 2022-2026 (disertakan di ZIP)
    2. Tabel Harga Berdasarkan Komoditas.xlsx — data terbaru dari user
    Kedua sumber digabung; data Excel akan menimpa seed untuk tanggal yang sama.
    """
    frames = []

    # ─── Sumber 1: Seed CSV historis ────────────────────────────────────────
    if os.path.exists(path_seed):
        df_seed = pd.read_csv(path_seed)
        df_seed['tanggal'] = pd.to_datetime(df_seed['tanggal'])
        df_seed = df_seed[['tanggal', 'harga']].dropna()
        frames.append(df_seed)
        print(f"✔ Seed historis: {len(df_seed)} baris "
              f"({df_seed['tanggal'].min().date()} – {df_seed['tanggal'].max().date()})")
    else:
        print("⚠ Seed historis tidak ditemukan. Hanya menggunakan data Excel.")

    # ─── Sumber 2: Akumulasi dari semua upload admin ────────────────────────
    akum_path = os.path.join(folder_data, 'Data_Harga_Bali_Akumulasi.csv')
    if os.path.exists(akum_path):
        df_akum = pd.read_csv(akum_path)
        df_akum['tanggal'] = pd.to_datetime(df_akum['tanggal'])
        df_akum = df_akum[['tanggal', 'harga']].dropna()
        if not df_akum.empty:
            frames.append(df_akum)
            print(f"✔ Data akumulasi admin: {len(df_akum)} baris "
                  f"({df_akum['tanggal'].min().date()} – {df_akum['tanggal'].max().date()})")

    # ─── Sumber 3: Excel dari user (fallback jika belum ada akumulasi) ─────
    if os.path.exists(path_harga):
        try:
            df_harga = pd.read_excel(path_harga)

            # Filter baris Bali
            df_bali = df_harga[df_harga['Komoditas (Rp)'].astype(str).str.strip() == 'Bali'].copy()
            if df_bali.empty:
                df_bali = df_harga.iloc[1:2].copy()  # fallback baris ke-2

            id_cols = ['No', 'Komoditas (Rp)']
            df_long = df_bali.melt(id_vars=id_cols, var_name='Tanggal_str', value_name='Harga')

            df_long['Tanggal_str'] = df_long['Tanggal_str'].astype(str).str.replace(' ', '')
            df_long['tanggal'] = pd.to_datetime(df_long['Tanggal_str'],
                                                 format='%d/%m/%Y', errors='coerce')
            df_long['harga_raw'] = (df_long['Harga'].astype(str)
                                    .str.replace(',', '').str.replace(' ', ''))
            df_long['harga'] = pd.to_numeric(df_long['harga_raw'], errors='coerce')
            df_excel = df_long.dropna(subset=['tanggal', 'harga'])[['tanggal', 'harga']]
            df_excel = df_excel[df_excel['harga'] > 0]

            if not df_excel.empty:
                frames.append(df_excel)
                print(f"✔ Data Excel: {len(df_excel)} baris "
                      f"({df_excel['tanggal'].min().date()} – {df_excel['tanggal'].max().date()})")
            else:
                print("⚠ Tidak ada data valid dari Excel.")
        except Exception as e:
            print(f"⚠ Gagal baca Excel: {e}")
    else:
        print("⚠ File Excel tidak ditemukan.")

    if not frames:
        print("❌ Tidak ada sumber data yang tersedia.")
        return None

    # ─── Gabung dan deduplicate (Excel menimpa seed untuk tanggal yang sama)
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sort_values('tanggal')
    df_all = df_all.drop_duplicates(subset='tanggal', keep='last')
    df_all = df_all.reset_index(drop=True)

    print(f"✔ Data gabungan: {len(df_all)} baris "
          f"({df_all['tanggal'].min().date()} – {df_all['tanggal'].max().date()})")
    return df_all


def proses_data_cabai():
    print("--- Memulai Proses Pengolahan Data ---")

    folder_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    path_harga  = os.path.join(folder_data, 'Tabel Harga Berdasarkan Komoditas.xlsx')
    path_seed   = os.path.join(folder_data, 'Data_Historis_Bali_Seed.csv')
    path_output = os.path.join(folder_data, 'data_cabai_siap_ai.csv')
    path_raya   = os.path.join(folder_data, 'Data_Hari_Raya.xlsx')

    # 1. Baca dan gabung semua sumber harga
    df_base = _baca_dan_gabung_sumber(folder_data, path_harga, path_seed)
    if df_base is None or len(df_base) < 50:
        print("❌ Data tidak cukup untuk diproses.")
        return None

    df_long = df_base.copy()

    # Interpolasi nilai kosong/nol
    df_long['harga'] = (df_long['harga']
                        .replace(0, np.nan)
                        .interpolate(method='linear')
                        .ffill().bfill())

    # ─── 2. Feature Engineering ─────────────────────────────────────────────
    df_long['hari_dalam_seminggu'] = df_long['tanggal'].dt.dayofweek
    df_long['bulan']               = df_long['tanggal'].dt.month
    df_long['kuartal']             = df_long['tanggal'].dt.quarter
    df_long['hari_dalam_bulan']    = df_long['tanggal'].dt.day

    # Lag features — hanya data masa lalu
    df_long['lag_1']  = df_long['harga'].shift(1)
    df_long['lag_2']  = df_long['harga'].shift(2)
    df_long['lag_3']  = df_long['harga'].shift(3)
    df_long['lag_7']  = df_long['harga'].shift(7)
    df_long['lag_14'] = df_long['harga'].shift(14)
    df_long['lag_21'] = df_long['harga'].shift(21)

    # Rolling stats — shift(1) agar TIDAK menyertakan harga hari ini (mencegah leakage)
    h_lag = df_long['harga'].shift(1)
    df_long['rolling_mean_7']  = h_lag.rolling(7).mean()
    df_long['rolling_mean_14'] = h_lag.rolling(14).mean()
    df_long['rolling_mean_30'] = h_lag.rolling(30).mean()
    df_long['rolling_std_7']   = h_lag.rolling(7).std()
    df_long['rolling_std_14']  = h_lag.rolling(14).std()

    # Selisih & Rasio — hanya lag, bukan harga hari ini
    df_long['selisih_1hari']   = df_long['lag_1'] - df_long['lag_2']
    df_long['selisih_7hari']   = df_long['lag_1'] - df_long['lag_7']
    df_long['rasio_vs_mean30'] = df_long['lag_1'] / df_long['rolling_mean_30'].replace(0, np.nan)

    # ─── 3. Kalender Hari Raya ──────────────────────────────────────────────
    try:
        df_raya = pd.read_excel(path_raya)
        df_raya['tanggal'] = pd.to_datetime(df_raya['Tanggal'])
        hr_dates = df_raya['tanggal'].tolist()
    except Exception:
        hr_dates = []
        print("⚠ Data hari raya tidak ditemukan, fitur fase akan bernilai 'normal' semua.")

    def get_proximity(tgl, hr_list):
        if not hr_list:
            return 999, 999
        diffs = [(tgl - hr).days for hr in hr_list]
        masa_depan = [d for d in diffs if d <= 0]
        masa_lalu  = [d for d in diffs if d > 0]
        hari_ke    = abs(min(masa_depan, key=abs)) if masa_depan else 999
        hari_sejak = min(masa_lalu)                if masa_lalu  else 999
        return hari_ke, hari_sejak

    def get_fase(tgl, hr_list):
        if not hr_list:
            return 'normal'
        diffs     = [(tgl - hr).days for hr in hr_list]
        mendatang = [-d for d in diffs if d <= 0]
        if not mendatang:
            return 'normal'
        d = min(mendatang)
        if   d <= 2:  return 'puncak'
        elif d <= 5:  return 'persiapan'
        elif d <= 10: return 'pasca'
        return 'normal'

    prox = df_long['tanggal'].apply(lambda t: get_proximity(t, hr_dates))
    df_long['hari_ke_hr']    = prox.apply(lambda x: x[0])
    df_long['hari_sejak_hr'] = prox.apply(lambda x: x[1])
    df_long['fase']          = df_long['tanggal'].apply(lambda t: get_fase(t, hr_dates))

    df_long['fase_normal']    = (df_long['fase'] == 'normal').astype(int)
    df_long['fase_pasca']     = (df_long['fase'] == 'pasca').astype(int)
    df_long['fase_persiapan'] = (df_long['fase'] == 'persiapan').astype(int)
    df_long['fase_puncak']    = (df_long['fase'] == 'puncak').astype(int)

    # ─── 4. Simpan ─────────────────────────────────────────────────────────
    df_final = df_long.dropna().reset_index(drop=True)
    print(f"✔ Dataset final: {len(df_final)} baris setelah hapus NaN dari lag/rolling")

    if len(df_final) < 100:
        print("⚠ PERINGATAN: Data kurang dari 100 baris — hasil model tidak akan optimal.")
        print("   Tambahkan file 'Data_Historis_Bali_Seed.csv' ke folder dataset/")

    df_final.to_csv(path_output, index=False)
    print(f"✔ Disimpan ke: {path_output}")
    return df_final


if __name__ == '__main__':
    proses_data_cabai()
