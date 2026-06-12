import os, sys, json, joblib
import pandas as pd
import numpy as np
from sklearn.ensemble      import RandomForestRegressor
from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from preprocess_data       import proses_data_cabai

BASE_DIR  = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

FITUR_COLS = [
    'hari_dalam_seminggu', 'bulan', 'kuartal', 'hari_dalam_bulan',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
    'rolling_std_7', 'rolling_std_14',
    'selisih_1hari', 'selisih_7hari', 'rasio_vs_mean30',
    'hari_ke_hr', 'hari_sejak_hr',
    'fase_normal', 'fase_pasca', 'fase_persiapan', 'fase_puncak'
]
FITUR_NUMERIK = [
    'hari_dalam_seminggu', 'bulan', 'kuartal', 'hari_dalam_bulan',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
    'rolling_std_7', 'rolling_std_14',
    'selisih_1hari', 'selisih_7hari', 'rasio_vs_mean30',
    'hari_ke_hr', 'hari_sejak_hr'
]
FITUR_BINER = [c for c in FITUR_COLS if c not in FITUR_NUMERIK]
TARGET = 'harga'

def _scale_features(X_train_raw, X_test_raw, scaler):
    """
    Normalisasi fitur numerik. Menggunakan pd.DataFrame constructor untuk
    menghindari bug assignment kolom di pandas 3.x pada Windows.
    """
    # Fit+transform data latih (numpy array)
    arr_train = scaler.fit_transform(X_train_raw[FITUR_NUMERIK].values.astype(float))
    arr_test  = scaler.transform(X_test_raw[FITUR_NUMERIK].values.astype(float))

    # Bangun DataFrame baru dengan nama kolom eksplisit — TIDAK assign in-place
    num_train = pd.DataFrame(arr_train, columns=FITUR_NUMERIK, index=X_train_raw.index)
    num_test  = pd.DataFrame(arr_test,  columns=FITUR_NUMERIK, index=X_test_raw.index)

    # Gabungkan kembali dengan fitur biner yang tidak diubah
    X_train_s = pd.concat([num_train, X_train_raw[FITUR_BINER].reset_index(drop=True).set_index(X_train_raw.index)], axis=1)[FITUR_COLS]
    X_test_s  = pd.concat([num_test,  X_test_raw[FITUR_BINER].reset_index(drop=True).set_index(X_test_raw.index)],  axis=1)[FITUR_COLS]
    return X_train_s, X_test_s

def latih_model():
    print("=== Memulai Pelatihan Model ===")

    df = proses_data_cabai()
    if df is None or len(df) < 50:
        print("❌ Data tidak cukup."); sys.exit(1)

    missing = [c for c in FITUR_COLS if c not in df.columns]
    if missing:
        print(f"❌ Kolom fitur hilang: {missing}"); sys.exit(1)

    # Split kronologis 80/20
    X = df[FITUR_COLS].copy()
    y = df[TARGET].copy()
    split_idx       = int(len(X) * 0.8)
    X_train_raw     = X.iloc[:split_idx].copy()
    X_test_raw      = X.iloc[split_idx:].copy()
    y_train         = y.iloc[:split_idx].copy()
    y_test          = y.iloc[split_idx:].copy()
    print(f"✔ Data latih: {len(X_train_raw)} | Data uji: {len(X_test_raw)}")

    # Normalisasi — robust untuk semua versi pandas
    scaler = MinMaxScaler()
    X_train, X_test = _scale_features(X_train_raw, X_test_raw, scaler)

    # Latih Random Forest
    print("🌲 Melatih Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=4,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    yp_rf  = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, yp_rf)
    rmse_rf= np.sqrt(mean_squared_error(y_test, yp_rf))
    mape_rf= float(np.mean(np.abs((y_test.values - yp_rf) /
                   np.where(y_test.values != 0, y_test.values, np.nan))) * 100)
    r2_rf  = r2_score(y_test, yp_rf)
    print(f"  RF → MAE: {mae_rf:,.0f} | RMSE: {rmse_rf:,.0f} | MAPE: {mape_rf:.2f}% | R²: {r2_rf:.4f}")

    # Latih Linear Regression
    print("📈 Melatih Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    yp_lr  = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, yp_lr)
    rmse_lr= np.sqrt(mean_squared_error(y_test, yp_lr))
    mape_lr= float(np.mean(np.abs((y_test.values - yp_lr) /
                   np.where(y_test.values != 0, y_test.values, np.nan))) * 100)
    r2_lr  = r2_score(y_test, yp_lr)
    print(f"  LR → MAE: {mae_lr:,.0f} | RMSE: {rmse_lr:,.0f} | MAPE: {mape_lr:.2f}% | R²: {r2_lr:.4f}")

    if min(mae_rf, mae_lr) < 1.0:
        print("⚠ PERINGATAN: MAE < Rp1 — kemungkinan masih ada data leakage!")

    # Simpan
    joblib.dump(rf,            os.path.join(MODEL_DIR, 'model_random_forest.pkl'))
    joblib.dump(lr,            os.path.join(MODEL_DIR, 'model_linear_regression.pkl'))
    joblib.dump(scaler,        os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(FITUR_COLS,    os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    joblib.dump(FITUR_NUMERIK, os.path.join(MODEL_DIR, 'numeric_columns.pkl'))
    joblib.dump(FITUR_BINER,   os.path.join(MODEL_DIR, 'binary_columns.pkl'))

    metrics = {
        'random_forest':     {'mae': round(mae_rf,2), 'rmse': round(rmse_rf,2),
                              'mape': round(mape_rf,2), 'r2': round(r2_rf,4),
                              'data_latih': len(X_train), 'data_uji': len(X_test)},
        'linear_regression': {'mae': round(mae_lr,2), 'rmse': round(rmse_lr,2),
                              'mape': round(mape_lr,2), 'r2': round(r2_lr,4),
                              'data_latih': len(X_train), 'data_uji': len(X_test)}
    }
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("✔ Semua model & metrik berhasil disimpan.")
    print("=== Pelatihan Selesai ===")
    return metrics

if __name__ == '__main__':
    latih_model()
