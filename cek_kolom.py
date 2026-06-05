import pandas as pd
import os

path_harga = os.path.join('dataset', 'Tabel Harga Berdasarkan Komoditas.xlsx')
df = pd.read_excel(path_harga)

print("--- DAFTAR KOLOM YANG DIKENALI PYTHON ---")
print(df.columns.tolist())  # Memunculkan semua kolom