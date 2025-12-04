import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_soil_moisture_dataset(n_rows=1000, seed: int = 0, save_csv: bool = False,
                                   path: str = 'app/data/soil_moisture_generated.csv', plot: bool = False,
                                   use_space_headers: bool = False,
                                   add_time: bool = False,
                                   n_locations: int = 1,
                                   start_date: str = '2024-01-01',
                                   period_days: int = 365,
                                   freq: str = 'D'):
    """Menghasilkan dataset sintetis dengan skema dan hubungan berikut:

    Kolom:
    - temperature: (Celsius)
    - humidity: (%)
    - rainfall: (mm)
    - cloud_cover: (%)
    - soil_moisture: (%) - target, diturunkan dari variabel lain

    Contoh baris:
    temperature humidity rainfall cloud_cover soil_moisture
    32 40 0 20 12
    28 60 10 80 35

    Parameter
    ----------
    n_rows : int
        Jumlah baris yang dihasilkan
    seed : int
        Seed acak untuk reproduktabilitas
    save_csv : bool
        Apakah menulis ke CSV (path)
    path : str
        Path keluaran untuk CSV
    plot : bool
        Jika True, simpan plot sebar rainfall vs soil_moisture

    Returns
    -------
    pandas.DataFrame
        DataFrame dengan skema yang dihasilkan
    """
    np.random.seed(seed)

    # Suhu: Normal sekitar 25C, dibatasi antara 5 dan 45
    temperature = np.random.normal(loc=25.0, scale=6.0, size=n_rows)
    temperature = np.clip(temperature, 5.0, 45.0).round(1)

    # Kelembaban: Normal sekitar 60%, dibatasi 5-100
    humidity = np.random.normal(loc=60.0, scale=15.0, size=n_rows)
    humidity = np.clip(humidity, 5.0, 100.0).round(1)

    # Hujan: Sebagian besar nilai kecil dengan kejadian besar sesekali (eksponensial)
    rainfall = np.random.exponential(scale=3.0, size=n_rows)
    # Sebagian hari kering (curah hujan = 0)
    dry_mask = np.random.rand(n_rows) < 0.35
    rainfall[dry_mask] = 0.0
    rainfall = np.round(rainfall, 1)

    # Tutup awan: 0-100 uniform
    cloud_cover = np.random.uniform(0, 100, n_rows).round(1)

    # Buat proxy kelembaban tanah sebagai kombinasi fitur
    # Normalisasi input ke rentang 0-1, menggunakan nilai min/maks tetap
    # Suhu berkaitan negatif dengan kelembaban; hujan & kelembaban berkaitan positif
    t_min, t_max = 5.0, 45.0
    h_min, h_max = 5.0, 100.0
    r_min, r_max = 0.0, np.max(rainfall) if np.max(rainfall) > 0 else 1.0
    c_min, c_max = 0.0, 100.0

    t_norm = (temperature - t_min) / (t_max - t_min)
    h_norm = (humidity - h_min) / (h_max - h_min)
    r_norm = (rainfall - r_min) / (r_max - r_min) if r_max != r_min else np.zeros_like(rainfall)
    c_norm = (cloud_cover - c_min) / (c_max - c_min)

    # Dasar kelembaban tanah: baseline 10%
    soil_norm = 0.12 + (0.55 * r_norm) + (0.20 * h_norm) - (0.15 * t_norm) + (0.05 * c_norm)
    # Tambahkan sedikit noise dan batasi antara 0..1
    soil_norm += np.random.normal(0.0, 0.03, size=n_rows)
    soil_norm = np.clip(soil_norm, 0.0, 1.0)
    soil_moisture = np.round(soil_norm * 100, 1)

    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall,
        'cloud_cover': cloud_cover,
        'soil_moisture': soil_moisture
    })
    if use_space_headers:
        df.columns = ['temperature', 'humidity', 'rainfall', 'cloud cover', 'soil moisture']

    # Jika add_time, perluas ke beberapa lokasi dan buat deret waktu
    if add_time:
        # Bangun panjang deret waktu per lokasi (period_days), replikasi per lokasi
        rows_per_location = max(1, period_days)
        total_rows = rows_per_location * n_locations
        # Hasilkan koordinat lokasi dan properti tekstur tanah
        latitudes = np.random.uniform(48.0, 55.0, n_locations)
        longitudes = np.random.uniform(6.0, 13.0, n_locations)
        # Fraksi tanah acak (jumlah 100)
        clay = np.random.uniform(2.0, 50.0, n_locations).round(1)
        sand = np.random.uniform(10.0, 90.0, n_locations).round(1)
        silt = np.clip(100.0 - (clay + sand), 0.0, 100.0).round(1)

        # Buat rentang waktu
        date_range = pd.date_range(start=start_date, periods=rows_per_location, freq=freq)

        rows = []
        for li in range(n_locations):
            # Sedikit modifikasi array dasar untuk tiap lokasi untuk mensimulasikan lokalitas
            temp_loc = np.clip(temperature[:rows_per_location] + np.random.normal(0.0, 2.0, rows_per_location), 5.0, 45.0).round(1)
            hum_loc = np.clip(humidity[:rows_per_location] + np.random.normal(0.0, 4.0, rows_per_location), 5.0, 100.0).round(1)
            rain_loc = np.clip(rainfall[:rows_per_location] + np.random.normal(0.0, 1.0, rows_per_location), 0.0, None).round(1)
            cloud_loc = np.clip(cloud_cover[:rows_per_location] + np.random.normal(0.0, 10.0, rows_per_location), 0.0, 100.0).round(1)
            
            # Hitung ulang kelembaban tanah per lokasi
            r_max_local = max(rain_loc.max(), 1.0)
            r_norm_loc = (rain_loc - r_min) / (r_max_local - r_min) if r_max_local != r_min else np.zeros_like(rain_loc)
            t_norm_loc = (temp_loc - t_min) / (t_max - t_min)
            h_norm_loc = (hum_loc - h_min) / (h_max - h_min)
            c_norm_loc = (cloud_loc - c_min) / (c_max - c_min)
            soil_norm_loc = 0.12 + (0.55 * r_norm_loc) + (0.20 * h_norm_loc) - (0.15 * t_norm_loc) + (0.05 * c_norm_loc)
            soil_norm_loc += np.random.normal(0.0, 0.03, size=rows_per_location)
            soil_norm_loc = np.clip(soil_norm_loc, 0.0, 1.0)
            soil_moisture_loc = np.round(soil_norm_loc * 100, 1)

            for i in range(rows_per_location):
                row = {
                    'time': date_range[i],
                    'latitude': latitudes[li],
                    'longitude': longitudes[li],
                    'clay_content': clay[li],
                    'sand_content': sand[li],
                    'silt_content': silt[li],
                    'sm_aux': max(0.0, soil_moisture_loc[i] + np.random.normal(0.0, 2.0)),
                    'sm_tgt': soil_moisture_loc[i]
                }
                rows.append(row)

        timeseries_df = pd.DataFrame(rows)
        df_ts = timeseries_df
        
        if save_csv:
            df_ts.to_csv(path, index=False)
    else:
        if save_csv:
            df.to_csv(path, index=False)

    return df


def create_data(samples=100, plot=True, save_csv=False, path='app/data/soil_moisture_generated.csv'):
    """Compatibility wrapper that returns (X, y) arrays.

    X: array(N, features=4) [temperature, humidity, rainfall, cloud_cover]
    y: array(N,) soil_moisture (0-100)
    """
    df = generate_soil_moisture_dataset(n_rows=samples, seed=0, save_csv=save_csv, path=path, plot=plot)
    features = ['temperature', 'humidity', 'rainfall', 'cloud_cover']
    X = df[features].values.astype(float)
    y = df['soil_moisture'].values.astype(float)
    return X, y


if __name__ == '__main__':
    # Simple generator demo
    df = generate_soil_moisture_dataset(n_rows=10000, seed=1, save_csv=True, path='app/data/soil_moisture_generated.csv', plot=True)
    print(df.head(6).to_string(index=False))