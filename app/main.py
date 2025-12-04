import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.neuralnetwork import NeuralNetwork
from app.function.layer import Dense
from app.function.activations import ReLU, Softmax, Linear, Sigmoid
from app.function.regularization import BatchNormalization, Dropout
from app.function.check_loss import CategoricalCrossentropy, MeanSquaredError
from app.function.metrics import calculate_accuracy, calculate_r2_score, calculate_mae, calculate_rmse
from app.data.generate_dataset import create_data

# Min-Max Scaling Normalization
def minmax_scale_np(arr, minv=None, maxv=None):
    arr = np.array(arr, dtype=float)
    if minv is None:
        minv = arr.min(axis=0)
    if maxv is None:
        maxv = arr.max(axis=0)
    denom = np.where((maxv - minv) == 0.0, 1.0, (maxv - minv))
    scaled = (arr - minv) / denom
    return scaled, minv, maxv

# Discretize continuous target into classes
def discretize_target(y, n_bins=5):
    percentiles = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    labels = np.digitize(y, percentiles[1:-1])
    return labels, percentiles


def main(dataset_path='app/data/soil_moisture_level.csv', epochs=500, batch_size=128, lr=0.001, regression=True):
    print('Starting Neural Network Training Pipeline')
    
    df = pd.read_csv(dataset_path)
    print(f'Dataset loaded: {len(df)} samples')
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'ttime' in df.columns and 'time' not in df.columns:
        df.rename(columns={'ttime': 'time'}, inplace=True)
    
    # Feature selection: 7 features (ammonia removed)
    feature_cols = ['pm1', 'pm2', 'pm3', 'luminosity', 'temperature', 'humidity', 'pressure']
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col = 'soil_moisture'
    
    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col} not found')
    
    print(f'Features: {feature_cols}')
    print(f'Target: {target_col}')
    
    # Extract and normalize data
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    Y = df[target_col].values.astype(np.float32)
    
    print(f'X shape: {X.shape}, Y shape: {Y.shape}')
    print(f'Y range: [{Y.min():.2f}, {Y.max():.2f}]')
    
    # Min-Max Normalization
    X_norm, x_min, x_max = minmax_scale_np(X)
    Y_norm = (Y - Y.min()) / (Y.max() - Y.min())
    
    print(f'Data normalized (Min-Max Scaling)')
    print('='*60)
    
    # Train-test split (80-20)
    n_train = int(0.8 * len(X_norm))
    indices = np.random.permutation(len(X_norm))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train = X_norm[train_idx]
    Y_train = Y_norm[train_idx]
    X_test = X_norm[test_idx]
    Y_test = Y_norm[test_idx]
    
    print(f'Train set: {len(X_train)} samples')
    print(f'Test set: {len(X_test)} samples')
    
    # Build and train model
    input_size = X_train.shape[1]
    model = NeuralNetwork()
    
    model.add(Dense(input_size, 32, learning_rate=lr))
    model.add(BatchNormalization(learning_rate=lr))
    model.add(ReLU())
    model.add(Dropout(0.2))
    
    model.add(Dense(32, 1, learning_rate=lr))
    model.add(Sigmoid())
    
    model.set_loss(MeanSquaredError())
    
    print('Model Architecture:')
    print(f'  Input: {input_size} neurons')
    print(f'  Hidden Layer: 32 neurons + ReLU + Dropout(0.2) + BatchNorm')
    print(f'  Output: 1 neuron + Sigmoid activation (bounded [0, 1])')
    print(f'Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}')
    print('='*60)
    
    model.train(X_train, Y_train.reshape(-1, 1), epochs=epochs, batch_size=batch_size)
    
    # Evaluate on test set
    pred_test = model.predict_proba(X_test).flatten()
    mae_test = np.mean(np.abs(pred_test - Y_test))
    
    print(f'Test MAE (normalized): {mae_test:.4f}')
    print(f'Test MAE (original): {mae_test * (Y.max() - Y.min()):.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Network for Soil Moisture Prediction')
    parser.add_argument('--dataset', default='app/data/soil_moisture_level.csv', help='Dataset path')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    main(dataset_path=args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate)
