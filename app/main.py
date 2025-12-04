import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.neuralnetwork import NeuralNetwork
from app.function.layer import Dense
from app.function.activations import ReLU, Softmax, Linear
from app.function.regularization import BatchNormalization, Dropout
from app.function.check_loss import CategoricalCrossentropy, MeanSquaredError
from app.function.metrics import calculate_accuracy
from app.data.dataset import create_data, generate_soil_moisture_dataset

# Normalisasi min-max untuk array numpy
def _minmax_scale_np(arr, minv=None, maxv=None):
    arr = np.array(arr, dtype=float)
    if minv is None:
        minv = arr.min(axis=0)
    if maxv is None:
        maxv = arr.max(axis=0)
    denom = np.where((maxv - minv) == 0.0, 1.0, (maxv - minv))
    scaled = (arr - minv) / denom
    return scaled, minv, maxv

# Diskretisasi target untuk klasifikasi
def discretize_target(y, n_bins=5):
    percentiles = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    labels = np.digitize(y, percentiles[1:-1])
    return labels, percentiles

# Membuat urutan data time series yang di-flattenkan
def create_sequences_flattened(df, features, target, seq_len):
    Xf = []
    Yf = []
    for loc, group in df.groupby('location_id'):
        group = group.sort_values('time') if 'time' in group else group
        feat_arr = group[features].values
        target_arr = group[target].values
        for i in range(len(feat_arr) - seq_len):
            win = feat_arr[i:i + seq_len]
            Xf.append(win.flatten())
            Yf.append(target_arr[i + seq_len])
    Xf = np.array(Xf, dtype=float)
    Yf = np.array(Yf, dtype=float)
    return Xf, Yf

# Melatih dan mengevaluasi model
def train_and_eval_model(X, Y, is_regression=False, n_classes=5, epochs=100, batch_size=64, lr=0.005):
    # Membangun model berdasarkan arsitektur default sederhana yang mirip dengan aplikasi lainnya
    input_size = X.shape[1]
    model = NeuralNetwork()
    model.add(Dense(input_size, 128, learning_rate=lr))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    model.add(Dense(128, 64, learning_rate=lr))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    if not is_regression:
        # Memastikan Y dalam bentuk label integer untuk klasifikasi
        Y_arr = np.array(Y)
        if Y_arr.ndim > 1:
            Y_flat = Y_arr.flatten()
        else:
            Y_flat = Y_arr
        # Jika float atau banyak nilai unik -> diskretisasi menjadi n_classes
        if Y_flat.dtype.kind in 'fc' or len(np.unique(Y_flat)) > n_classes:
            Y_classes, _ = discretize_target(Y_flat, n_bins=n_classes)
        else:
            Y_classes = Y_flat.astype(int)
        model.add(Dense(64, n_classes, learning_rate=lr))
        model.add(Softmax())
        model.set_loss(CategoricalCrossentropy(regularization_l2=1e-4))
        model.train(X, Y_classes, epochs=epochs, batch_size=batch_size)
        preds = model.predict_proba(X)
        acc = calculate_accuracy(Y_classes, preds)
        return model, acc, None
    else:
        model.add(Dense(64, 1, learning_rate=lr))
        model.add(Linear())
        model.set_loss(MeanSquaredError())
        model.train(X, Y.reshape(-1, 1), epochs=epochs, batch_size=batch_size)
        preds = model.predict_proba(X).flatten()
        mae = np.mean(np.abs(preds - Y.flatten()))
        return model, mae, None


def main(dataset_path='app/data/soil_moisture.csv', use_timeseries=False, generate=False, n_locations=5, period_days=365, seq_length=15, n_classes=5, epochs=100, batch_size=64, lr=0.005, regression=False):
    print('Starting pipeline...')
    if generate:
        print('Generating synthetic time series dataset...')
        df = generate_soil_moisture_dataset(n_rows=period_days, seed=42, save_csv=True, path=dataset_path, add_time=True, n_locations=n_locations, period_days=period_days)
    elif os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        # Use inline create_data wrapper for simple synthetic tabular data
        X, Y = create_data(samples=1000)
        is_reg = regression
        model, metric, _ = train_and_eval_model(X, Y, is_reg, n_classes=n_classes, epochs=epochs, batch_size=batch_size, lr=lr)
        print('Done. Metric:', metric)
        return

    # Normalize columns
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    # For timeseries: generate location_id if not present
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['location_id'] = df.groupby(['latitude', 'longitude']).ngroup()
    else:
        df['location_id'] = 0

    # Choose features depending on columns present
    if 'temperature' in df.columns and 'soil_moisture' in df.columns:
        feature_cols = [c for c in ['temperature', 'humidity', 'rainfall', 'cloud_cover'] if c in df.columns]
        target_col = 'soil_moisture'
        if 'sm_tgt' not in df.columns:
            df['sm_tgt'] = df['soil_moisture']
    else:
        feature_cols = [c for c in ['latitude', 'longitude', 'clay_content', 'sand_content', 'silt_content', 'sm_aux'] if c in df.columns]
        target_col = 'sm_tgt'

    # Split june holdout if time present
    if 'time' in df.columns:
        june_df = df[df['time'].dt.month == 6]
        train_df = df[df['time'].dt.month != 6]
    else:
        june_df = pd.DataFrame()
        train_df = df

    if len(train_df) == 0:
        raise RuntimeError('No training data')

    to_scale = feature_cols + [target_col]
    train_scaled, minv, maxv = _minmax_scale_np(train_df[to_scale].values)
    train_df_scaled = train_df.copy()
    train_df_scaled[to_scale] = train_scaled
    if len(june_df) > 0:
        june_scaled = (june_df[to_scale].values - minv) / np.where((maxv - minv) == 0, 1, (maxv - minv))
        june_df_scaled = june_df.copy()
        june_df_scaled[to_scale] = june_scaled
    else:
        june_df_scaled = pd.DataFrame()

    # Filter location counts
    counts = train_df_scaled.groupby('location_id').size()
    exclude = counts[counts < seq_length].index
    train_df_scaled = train_df_scaled[~train_df_scaled['location_id'].isin(exclude)].reset_index(drop=True)
    if len(june_df_scaled) > 0:
        june_df_scaled = june_df_scaled[~june_df_scaled['location_id'].isin(exclude)].reset_index(drop=True)

    X_seq, Y_seq = create_sequences_flattened(train_df_scaled, feature_cols, target_col, seq_length)
    if X_seq.size == 0:
        raise RuntimeError('No sequences generated')

    if not regression:
        Y_classes, edges = discretize_target(Y_seq, n_bins=n_classes)
        model, metric, _ = train_and_eval_model(X_seq, Y_classes, is_regression=False, n_classes=n_classes, epochs=epochs, batch_size=batch_size, lr=lr)
        print('Train accuracy:', metric)
    else:
        model, metric, _ = train_and_eval_model(X_seq, Y_seq, is_regression=True, n_classes=n_classes, epochs=epochs, batch_size=batch_size, lr=lr)
        print('Train MAE:', metric)

    if len(june_df_scaled) > 0:
        X_june, Y_june = create_sequences_flattened(june_df_scaled, feature_cols, target_col, seq_length)
        if X_june.size != 0:
            if not regression:
                Y_june_classes = np.digitize(Y_june, edges[1:-1])
                preds = model.predict_proba(X_june)
                acc = calculate_accuracy(Y_june_classes, preds)
                print('June holdout accuracy:', acc)
            else:
                preds = model.predict_proba(X_june).flatten()
                tgt_min = minv[-1]
                tgt_max = maxv[-1]
                denom = tgt_max - tgt_min if (tgt_max - tgt_min) != 0 else 1.0
                preds_original = preds * denom + tgt_min
                y_original = Y_june.flatten() * denom + tgt_min
                mae = np.mean(np.abs(preds_original - y_original))
                print('June holdout MAE:', mae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='app/data/soil_moisture.csv')
    parser.add_argument('--seq_length', type=int, default=15)
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--n_locations', type=int, default=5)
    parser.add_argument('--period_days', type=int, default=365)
    args = parser.parse_args()

    main(dataset_path=args.dataset, use_timeseries=True, generate=args.generate, n_locations=args.n_locations, period_days=args.period_days, seq_length=args.seq_length, n_classes=args.n_classes, epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate, regression=args.regression)
