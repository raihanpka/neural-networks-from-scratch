# Project Based Learning Neural Network untuk Prediksi Soil Moisture

Implementasi deep learning untuk memprediksi tingkat kelembaban tanah menggunakan data sensor lingkungan. Dibangun dengan arsitektur single hidden layer (sesuai constraint) dan menampilkan backpropagation manual, normalisasi, dan teknik regularisasi.

## Dataset

**Sumber**: `app/data/soil_moisture.csv` [Kaggle](https://www.kaggle.com/datasets/jshndeep/soil-moisture-levels)

- **Total Sampel**: 20.166 records
- **Target Variable**: soil_moisture (range: 0-7937 unit arbitrary)
- **Features** (7 input setelah preprocessing):
    - PM1: Partikulat matter 1 mikron
    - PM2: Partikulat matter 2,5 mikron
    - PM3: Partikulat matter 10 mikron
    - Luminosity: Pengukuran intensitas cahaya
    - Temperature: Suhu lingkungan
    - Humidity: Persentase kelembaban relatif
    - Pressure: Tekanan atmosfer

**Feature yang Dikecualikan**:
- Ammonia: tidak digunakan karena masalah kualitas data (78,6% nilai nol, 21,4% outlier dengan korelasi -0,43)

## Data Preprocessing

Pipeline preprocessing mengikuti urutan ini:

1. **Feature Selection**: Hapus ammonia (kualitas rendah), pertahankan 7 sensor feature
2. **Data Cleaning**: Tangani nilai NaN dengan mengganti menjadi 0
3. **Train-Test Split**: 70% training (16.132 sampel), 30% test (4.034 sampel) dengan permutasi acak
4. **Normalisasi**: Terapkan Min-Max scaling (detail di bawah)

## Metode Normalisasi

Formula Min-Max Scaling: 
$$x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Proses**:
1. Hitung min dan max dari training set saja
2. Terapkan parameter scaling yang sama ke test set (mencegah data leakage)
3. Normalisasi target variable (Y) secara terpisah menggunakan batas training set-nya
4. Hasil: Semua feature dan target di-scale ke range [0, 1]

**Denormalisasi** (untuk interpretasi prediksi):
$$x_{original} = x_{normalized} \times (x_{max} - x_{min}) + x_{min}$$

## Arsitektur Model

Neural network single hidden layer dengan regularisasi:

```
Input Layer (7 neuron)
        ↓
Dense Layer (32 neuron, learning_rate=0.001)
        ↓
BatchNormalization (learnable parameter: gamma, beta)
        ↓
ReLU Activation (max(0, x))
        ↓
Dropout (probability=0.2)
        ↓
Dense Layer (1 neuron, learning_rate=0.001)
        ↓
Sigmoid Activation (output bounded [0, 1] yang kemudian di-denormalisasi ke [0, 7937] dari dataset)
```

**Pilihan Arsitektur**:
- **Single Hidden Layer**: Sesuai project constraint, dipilih 32 neuron sebagai trade-off optimal antara kapasitas dan pencegahan overfitting
- **ReLU Activation**: Transformasi non-linear untuk hidden layer, memungkinkan pembelajaran pola kompleks
- **Sigmoid Output**: Menghasilkan prediksi terbatas pada range [0, 1], setelah denormalisasi menghasilkan prediksi valid dalam range target asli [0, 7937]. Mencegah prediksi negatif atau exceeding maksimum nilai.
- **BatchNormalization**: Normalisasi input layer, stabilisasi training, kurangi internal covariate shift
- **Dropout**: Menonaktifkan beberapa neuron secara acak saat training, kurangi co-adaptation dan overfitting

## Konfigurasi Training

- **Loss Function**: Mean Absolute Error (MAE) 
- **Optimizer**: Gradient Descent dengan Backpropagation
- **Learning Rate**: 0.001 (kontrol ukuran langkah dalam update parameter)
- **Batch Size**: 128 sampel per gradient update
- **Epochs**: 500 pass lengkap melalui training data
- **Gradient Clipping**: ±1.0 (cegah exploding gradient)
- **Output Activation**: Sigmoid (untuk bounded output)

### Output Activation: Sigmoid

Model menggunakan **Sigmoid activation** pada output layer untuk menjamin prediksi berada dalam range yang valid:

**Fungsi Sigmoid**: 
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
Range output: [0, 1]

**Alasan Penggunaan Sigmoid**:
- Mencegah prediksi negatif (yang tidak valid untuk soil_moisture)
- Mencegah prediksi exceeding maksimum nilai target (7937)
- Memberikan probabilitas-like output yang lebih stabil saat training
- Menghindari gradient vanishing pada nilai ekstrem

**Pipeline Denormalisasi**:
1. Sigmoid output bounded: $\hat{y}_{sigmoid} \in [0, 1]$
2. Denormalisasi ke range target asli: 
    $$\hat{y}_{original} = \hat{y}_{sigmoid} \times (y_{max} - y_{min}) + y_{min}$$
3. Hasil: $\hat{y}_{original} \in [0, 7937]$ (always valid range)

**Proses Gradient Descent**:
1. Forward pass: Propagasi input melalui layer jaringan
2. Hitung loss: MSE antara prediksi dan target
3. Backward pass: Hitung gradient via chain rule
4. Clip gradient: Pastikan |gradient| ≤ 1.0
5. Update parameter: weights ← weights - learning_rate × gradients
6. Update BatchNorm: parameter gamma, beta diupdate dengan learning rate yang sama

## Metrik Performa

### Metrik Penilaian

1. **R² Score (Coefficient of Determination)**
     - Formula: $R^2 = 1 - \frac{\sum(y_{true} - y_{pred})^2}{\sum(y_{true} - y_{mean})^2}$
     - Range: -∞ hingga 1.0 (1.0 adalah prediksi sempurna)
     - Interpretasi: Proporsi varians target yang dijelaskan model
     - **Training R²**: 0.8085 (good - menjelaskan 80.85% varians training)
     - **Test R²**: 0.7765 (generalisasi solid - menjelaskan 77.65% varians test)

2. **Mean Absolute Error (MAE)**
     - Formula: $MAE = \frac{1}{n} \sum |y_{true} - y_{pred}|$
     - **Test MAE**: 12.60% (persentase target range)
     - **Test MAE (unit asli)**: ~996 unit
     - Interpretasi: Deviasi absolut rata-rata dari nilai sebenarnya

3. **Root Mean Squared Error (RMSE)**
     - Formula: $RMSE = \sqrt{\frac{1}{n} \sum (y_{true} - y_{pred})^2}$
     - Penekanan pada error besar lebih dari MAE
     - Dilaporkan sebagai persentase target range untuk interpretabilitas

## Ringkasan Hasil

| Metrik | Train | Test | Status |
|--------|-------|------|--------|
| R² Score | 0.8085 | 0.7765 | Kuat |
| MAE (%) | ~9.03% | 9.85% | Cukup |
| RMSE (%) | ~14.32% | 16.97% | Cukup |

**Interpretasi**: Model mencapai performa prediktif kuat pada test data unseen dengan R² > 0.77, menunjukkan arsitektur 7-feature single-layer efektif menangkap pola kelembaban tanah meskipun ada constraint arsitektur.`

## Detail Implementasi

### File

- `app/main.py`: CLI training pipeline
- `app/web/streamlit.py`: Web interface interaktif
- `app/models/neuralnetwork.py`: Core neural network class
- `app/function/layer.py`: Dense layer dengan backpropagation
- `app/function/activations.py`: ReLU, Linear, Sigmoid, Tanh, Softmax
- `app/function/regularization.py`: BatchNormalization dan Dropout
- `app/function/optimizers.py`: Implementasi gradient descent
- `app/function/metrics.py`: Perhitungan R², MAE, RMSE, MAPE
- `app/data/dataset.py`: Utilitas data loading dan preprocessing

### Komponen Neural Network

1. **Forward Propagation**: Komputasi layer sekuensial dengan activation function
2. **Backpropagation**: Komputasi gradient chain rule untuk semua parameter
3. **Normalisasi**: Min-Max scaling dengan persistensi parameter
4. **Regularisasi**: BatchNormalization untuk stabilitas, Dropout untuk generalisasi
5. **Stabilitas Numerik**: Gradient clipping untuk cegah exploding gradient

## Catatan tentang Limitasi

1. **Single Hidden Layer Constraint**: Membatasi ekspresivitas model; ceiling performa sekitar R² = 0.80
2. **Manual Backpropagation**: Implementasi intentional untuk tujuan edukasi; tidak optimal untuk deployment skala besar
3. **Feature Dependencies**: Model dilatih hanya pada 7 feature saat ini; menambah/menghapus feature memerlukan retraining
4. **Ammonia Exclusion**: Keputusan preprocessing kritis berdasarkan analisis kualitas data; model dengan ammonia menunjukkan performa degraded

---

## Setup & Instalasi

### Requirements

- Python 3.10 atau lebih baru
- pip atau uv package manager

### Install Dependencies

Menggunakan pip:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Menggunakan uv (lebih cepat):
```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

## Penggunaan

### Training dengan Command Line

```bash
python app/main.py \
    --dataset app/data/soil_moisture.csv \
    --epochs 500 \
    --batch_size 128 \
    --learning_rate 0.001
```

**Argumen**:
- `--dataset`: Path file CSV (default: app/data/soil_moisture.csv)
- `--epochs`: Jumlah training epoch (default: 500)
- `--batch_size`: Sampel per gradient update (default: 128)
- `--learning_rate`: Learning rate untuk optimizer (default: 0.001)

### Web Interface Interaktif

```bash
streamlit run app/web/streamlit.py
```

Web interface menyediakan:
- Feature selection dan konfigurasi
- Pilihan metode normalisasi
- Tuning hyperparameter model
- Progress training real-time
- Evaluasi test set dengan visualisasi
- Analisis error (distribusi dan scatter plot)