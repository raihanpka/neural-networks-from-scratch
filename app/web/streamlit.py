import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.models.neuralnetwork import NeuralNetwork
from app.function.layer import Dense
from app.function.activations import ReLU, Softmax, Sigmoid, Tanh, Linear
from app.function.regularization import BatchNormalization, Dropout
from app.function.check_loss import CategoricalCrossentropy, MeanSquaredError, MeanAbsoluteError
from app.function.metrics import calculate_accuracy, calculate_r2_score, calculate_mae, calculate_rmse, calculate_mape

st.set_page_config(page_title="Group 1 - Neural Network (PBL AI)", layout="wide", page_icon="📊")

st.title("Soil Moisture Prediction - Neural Network Manual")

def normalize_data(data, method='minmax'):
    if method == 'minmax':
        return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)
    elif method == 'zscore':
        return (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    elif method == 'none':
        return data
    return data


def get_activation_class(name):
    activations = {
        'Linear': Linear,
        'Sigmoid': Sigmoid,
        'Tanh': Tanh,
        'Softmax': Softmax,
        'ReLU': ReLU,
    }
    return activations.get(name, ReLU)


def get_loss_class(name):
    losses = {
        'Categorical Crossentropy': CategoricalCrossentropy,
        'Mean Squared Error': MeanSquaredError,
        'Mean Absolute Error': MeanAbsoluteError
    }
    return losses.get(name, CategoricalCrossentropy)


def draw_neural_network(layer_sizes, layer_names=None, activations=None):
    """Visualisasi arsitektur neural network"""
    # Atur ukuran figure yang lebih sesuai
    fig, ax = plt.subplots(figsize=(6, 5))
    
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    
    layer_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    # Atur spacing yang lebih baik
    v_spacing = 1.2  # Jarak vertikal antar neuron
    h_spacing = 2.2  # Jarak horizontal antar layer
    
    # Atur margin untuk teks label
    text_margin = 1.0
    
    neuron_positions = {}
    
    for layer_idx, n_neurons in enumerate(layer_sizes):
        display_neurons = min(n_neurons, 6)
        show_ellipsis = n_neurons > 6
        
        layer_height = (display_neurons - 1) * v_spacing
        start_y = layer_height / 2
        
        x = layer_idx * h_spacing
        
        for neuron_idx in range(display_neurons):
            y = start_y - neuron_idx * v_spacing
            neuron_positions[(layer_idx, neuron_idx)] = (x, y)
            
            color = layer_colors[layer_idx % len(layer_colors)]
            circle = plt.Circle((x, y), 0.25, color=color, ec='white', linewidth=2, zorder=3)
            ax.add_patch(circle)
            if layer_idx == 0:
                ax.text(x, y, str(neuron_idx + 1), ha='center', va='center', 
                       fontsize=8, color='white', fontweight='bold', zorder=4)
        
        if show_ellipsis:
            y_ellipsis = start_y - display_neurons * v_spacing + v_spacing/2
            ax.text(x, y_ellipsis - 0.3, '...', ha='center', va='center', 
                   fontsize=16, color='#555', fontweight='bold')
            ax.text(x, y_ellipsis - 0.8, f'({n_neurons})', ha='center', va='center', 
                   fontsize=9, color='#777')
    
    for layer_idx in range(n_layers - 1):
        n_current = min(layer_sizes[layer_idx], 6)
        n_next = min(layer_sizes[layer_idx + 1], 6)
        
        for i in range(n_current):
            for j in range(n_next):
                start_pos = neuron_positions[(layer_idx, i)]
                end_pos = neuron_positions[(layer_idx + 1, j)]
                
                ax.plot([start_pos[0] + 0.2, end_pos[0] - 0.2], 
                       [start_pos[1], end_pos[1]], 
                       color='#cccccc', linewidth=0.7, alpha=0.6, zorder=1)
    
    for layer_idx, (name, size) in enumerate(zip(layer_names, layer_sizes)):
        x = layer_idx * h_spacing
        y_top = (min(size, 6) - 1) * v_spacing / 2 + 0.8
        
        label = f"{name}\n({size} neurons)"
        if activations and layer_idx > 0 and layer_idx <= len(activations):
            label += f"\n[{activations[layer_idx-1]}]"
        
        ax.text(x, y_top, label, ha='center', va='bottom', 
               fontsize=10, fontweight='bold', color='#333')
    # Atur batas plot dengan margin yang lebih baik
    x_margin = 0.5
    y_margin = 1.0
    ax.set_xlim(-x_margin, (n_layers - 1) * h_spacing + x_margin)
    max_display = min(max_neurons, 6)
    ax.set_ylim(-max_display * v_spacing * 0.5 - y_margin, 
                max_display * v_spacing * 0.5 + y_margin)
    ax.set_aspect('auto')  # Gunakan 'auto' untuk mencegah distorsi
    ax.axis('off')
    
    # Atur padding untuk mencegah teks terpotong
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    plt.tight_layout()
    return fig


def discretize_target(y, n_bins=5):
    """Diskritisasi target kontinyu menjadi kelas-kelas"""
    percentiles = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    labels = np.digitize(y, percentiles[1:-1])
    return labels, percentiles


DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/soil_moisture_level.csv')

with st.sidebar:
    st.header("Konfigurasi Data")

    # Pilih sumber dataset (mutually exclusive)
    data_source = st.radio("Sumber dataset", 
                           options=["Built-in Sensor Dataset", "Upload CSV"],
                           index=0)

    df = None

    # Built-in dataset options
    if data_source == "Built-in Sensor Dataset":
        st.subheader("Dataset Sensor Bawaan")
        if os.path.exists(DATA_PATH):
            if st.button("Muat Dataset Bawaan"):
                try:
                    df = pd.read_csv(DATA_PATH)
                    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
                    # map ttime to time if present
                    if 'ttime' in df.columns and 'time' not in df.columns:
                        df.rename(columns={'ttime': 'time'}, inplace=True)
                    st.session_state['df'] = df
                    # Determine built-in defaults based on header
                    cols = set(df.columns.tolist())
                    if {'pm1', 'pm2', 'pm3', 'ammonia', 'luminosity', 'temperature', 'humidity', 'pressure', 'soil_moisture'}.issubset(cols):
                        st.session_state['feature_cols'] = [c for c in ['pm1', 'pm2', 'pm3', 'ammonia', 'luminosity', 'temperature', 'humidity', 'pressure'] if c in df.columns]
                        st.session_state['target_col'] = 'soil_moisture'
                        st.session_state['builtin_schema'] = 'sensor'
                    else:
                        # fallback: pick numeric columns
                        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                        st.session_state['feature_cols'] = numeric_cols[:-1]
                        st.session_state['target_col'] = numeric_cols[-1] if numeric_cols else None
                    st.success(f"Dataset dimuat: {len(df)} sampel")
                except Exception as e:
                    st.error(f"Gagal memuat dataset bawaan: {e}")
        else:
            st.warning("Dataset bawaan tidak ditemukan di path: " + DATA_PATH)

    # Upload CSV options
    elif data_source == "Upload CSV":
        st.subheader("Upload file CSV")
        enable_upload = st.checkbox("Aktifkan Upload CSV", value=False)
        if enable_upload:
            uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state['df'] = df
                    st.success(f"Dataset loaded: {df.shape[0]} baris")
                except Exception as e:
                    st.error(f"Gagal membaca file CSV: {e}")
                    df = None
            else:
                st.info("Silakan pilih file CSV untuk di-upload")
        else:
            st.info("Aktifkan opsi upload untuk menampilkan uploader")

    # Jika df telah terisi (dari salah satu pilihan), tunjukkan opsi kolom/preprocessing
    if df is not None:
        st.subheader("Pilih Kolom")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        default_features = ['pm1', 'pm2', 'pm3', 'luminosity', 'temperature', 'humidity', 'pressure']
        default_features = [f for f in default_features if f in numeric_columns]

        feature_cols = st.multiselect(
            "Kolom Fitur (X)",
            options=numeric_columns,
            default=default_features if default_features else (numeric_columns[:-1] if len(numeric_columns) > 1 else numeric_columns),
            disabled=(data_source == "Built-in Sensor Dataset")
        )

        default_target = 'soil_moisture' if 'soil_moisture' in numeric_columns else (numeric_columns[-1] if numeric_columns else None)
        target_col = None
        if numeric_columns:
            target_col = st.selectbox(
                "Kolom Target (Y)",
                options=numeric_columns,
                index=numeric_columns.index(default_target) if default_target in numeric_columns else 0,
                disabled=(data_source == "Built-in Sensor Dataset")
            )

        n_classes = st.slider("Jumlah kelas (diskritisasi target)", 3, 10, 5,
                             help="Target kontinyu akan di-bin menjadi n kelas")
        
        if data_source == "Built-in Sensor Dataset":
            # Keep existing session values if present, otherwise set defaults
            st.session_state['regression_mode'] = st.session_state.get('regression_mode', True)
            st.session_state['soil_moisture_scale'] = st.session_state.get('soil_moisture_scale', 1000.0)

        st.subheader("Preprocessing")
        
        st.info("""
        Data Preprocessing:
        - Ammonia dihapus: 78.6% nilai 0, 21.4% outlier, korelasi lemah (-0.43)
        - Fitur aktif: PM1, PM2, PM3, Luminosity, Temperature, Humidity, Pressure (7 fitur)
        - Target: Soil Moisture (regression kontinyu, range 0-7937)
        """)
        
        regression_mode = st.session_state.get('regression_mode', False)
        
        normalize_method = st.selectbox(
            "Metode Normalisasi",
            options=['minmax', 'zscore', 'none'],
            format_func=lambda x: {
                'none': 'Tidak Ada',
                'minmax': 'Min-Max Scaling (0-1)',
                'zscore': 'Z-Score Standardization'
            }.get(x, x),
            index=0,
            help="Min-Max Scaling: (x-min)/(max-min). Formula: nilai dipetakan ke range [0,1]"
        )

        # Simpan ke session state
        st.session_state['df'] = df
        st.session_state['feature_cols'] = feature_cols
        st.session_state['target_col'] = target_col
        st.session_state['normalize_method'] = normalize_method
        st.session_state['n_classes'] = n_classes
        st.session_state['data_source'] = data_source

tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Arsitektur", "Training", "Hasil"])

with tab1:
    st.header("Eksplorasi Dataset Soil Moisture")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Jumlah Sampel", f"{df.shape[0]:,}")
        with col2:
            st.metric("Jumlah Fitur", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            # show average of the detected target column if available
            t_col = st.session_state.get('target_col', None)
            if t_col and t_col in df.columns:
                try:
                    st.metric("Rata-rata SM", f"{df[t_col].mean():.3f}")
                except Exception:
                    pass
        
        if data_source == "Built-in Sensor Dataset":
            st.subheader("Deskripsi Dataset")
            if 'df' in st.session_state:
                df_info = st.session_state['df']
                # Show default feature/target choice if set
                fc = st.session_state.get('feature_cols', [])
                tc = st.session_state.get('target_col', None)
                st.markdown(f"**Default Fitur:** {', '.join(fc)}")
                st.markdown(f"**Default Target:** {tc}")
        
        st.subheader("Sample Data")
        st.dataframe(df.head(15), width='stretch')
        
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe().T.style.format("{:.4f}"), width='stretch')
        
        st.subheader("Visualisasi Data")
        
        col1 = st.columns(1)[0]
        
        with col1:
            # Sensor-specific visualization: choose a sensor feature to compare with soil_moisture
            if 'soil_moisture' in df.columns:
                # Prefer pm1 if present, else luminosity, else temperature
                if 'pm1' in df.columns:
                    feature_x = 'pm1'
                    xlabel = 'PM1'
                elif 'luminosity' in df.columns:
                    feature_x = 'luminosity'
                    xlabel = 'Luminosity'
                elif 'temperature' in df.columns:
                    feature_x = 'temperature'
                    xlabel = 'Temperature'
                else:
                    feature_x = None

                if feature_x is not None and 'soil_moisture' in df.columns:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                    # Plot 1: PM1 vs Soil Moisture
                    s1 = axes[0].scatter(df[feature_x], df['soil_moisture'],
                                        c=df['soil_moisture'], cmap='YlOrBr',
                                        alpha=0.5, s=10)
                    axes[0].set_xlabel(xlabel)
                    axes[0].set_ylabel('Soil Moisture')
                    axes[0].set_title(f'{xlabel} vs Soil Moisture')
                    plt.colorbar(s1, ax=axes[0], label='Soil Moisture')
                    axes[0].grid(True, alpha=0.3)

                    # Plot 2: Luminosity vs Soil Moisture
                    s2 = axes[1].scatter(df['luminosity'], df['soil_moisture'],
                                        c=df['soil_moisture'], cmap='YlOrBr',
                                        alpha=0.5, s=10)
                    axes[1].set_xlabel('Luminosity')
                    axes[1].set_ylabel('Soil Moisture')
                    axes[1].set_title('Luminosity vs Soil Moisture')
                    plt.colorbar(s2, ax=axes[1], label='Soil Moisture')
                    axes[1].grid(True, alpha=0.3)

                    # Plot 3: Temperature vs Soil Moisture
                    s3 = axes[2].scatter(df['temperature'], df['soil_moisture'],
                                        c=df['soil_moisture'], cmap='coolwarm',
                                        alpha=0.5, s=10)
                    axes[2].set_xlabel('Temperature')
                    axes[2].set_ylabel('Soil Moisture')
                    axes[2].set_title('Temperature vs Soil Moisture')
                    plt.colorbar(s3, ax=axes[2], label='Soil Moisture')
                    axes[2].grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)
        
        feature_cols = st.session_state.get('feature_cols', [])
        if len(feature_cols) > 0:
            st.subheader("Distribusi Fitur")
            n_cols = min(3, len(feature_cols))
            n_rows = (len(feature_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            axes = np.atleast_2d(axes).flatten()
            
            for idx, col in enumerate(feature_cols):
                axes[idx].hist(df[col].dropna(), bins=30, alpha=0.7, 
                              color='steelblue', edgecolor='white')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frekuensi')
                axes[idx].set_title(f'Distribusi {col}')
                axes[idx].grid(True, alpha=0.3)
            
            for idx in range(len(feature_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("Silakan muat dataset di sidebar untuk memulai")

with tab2:
    st.header("Konfigurasi Arsitektur Neural Network")

    st.markdown("""
    Library yang digunakan untuk membangun model: 
    - Numpy
    - Pandas
    - Matplotlib & NetworkX (untuk visualisasi)
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Konfigurasi Layer")
        
        n_layers = st.number_input("Jumlah Hidden Layer", min_value=1, max_value=10, value=1)
        
        layer_config = []
        activations_list = []
        
        for i in range(n_layers):
            st.markdown(f"**Hidden Layer {i+1}**")
            col_a, col_b = st.columns(2)
            
            with col_a:
                neurons = st.number_input(
                    f"Neurons",
                    min_value=4, max_value=512, value=32,
                    key=f"neurons_{i}",
                    help="Untuk 1 hidden layer, coba 16-32 neurons"
                )
                activation = st.selectbox(
                    f"Aktivasi",
                    options=['Linear', 'Sigmoid', 'Tanh', 'Softmax', 'ReLU'],
                    index=4,
                    key=f"activation_{i}",
                    help="Umumnya menggunakan ReLU di hidden layer"
                )
            
            with col_b:
                use_batchnorm = st.checkbox(f"Batch Normalization", value=True, key=f"bn_{i}")
                dropout_rate = st.number_input(
                    f"Dropout Rate",
                    min_value=0.0, max_value=0.9, value=0.2,
                    step=0.05,
                    key=f"dropout_{i}",
                    help="Coba 0.2-0.3 (terlalu tinggi = underfitting)"
                )
            
            layer_config.append({
                'neurons': neurons,
                'activation': activation,
                'batchnorm': use_batchnorm,
                'dropout': dropout_rate
            })
            activations_list.append(activation)
        
        st.markdown("**Output Layer**")
        output_activation = st.selectbox(
            "Aktivasi Output",
            options=['Linear', 'Sigmoid', 'Softmax', 'Tanh', 'ReLU'],
            index=1,
        )
        activations_list.append(output_activation)
        
        st.session_state['layer_config'] = layer_config
        st.session_state['output_activation'] = output_activation
    
    with col2:
        st.subheader("Visualisasi Arsitektur")
        
        if 'df' in st.session_state and st.session_state.get('feature_cols'):
            input_size = len(st.session_state['feature_cols'])
        else:
            input_size = 6
        
        # Determine output size based on regression mode
        regression_mode = st.session_state.get('regression_mode', False)
        if regression_mode:
            output_size = 1
            output_label = "1 output"
        else:
            output_size = st.session_state.get('n_classes', 1)
            output_label = f"{output_size} kelas"
        
        layer_sizes = [input_size] + [l['neurons'] for l in layer_config] + [output_size]
        layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(layer_config))] + ['Output']
        
        fig = draw_neural_network(layer_sizes, layer_names, activations_list)
        st.pyplot(fig)
        
        st.markdown("**Ringkasan Arsitektur:**")
        total_params = 0
        prev_size = input_size
        for i, lc in enumerate(layer_config):
            params = prev_size * lc['neurons'] + lc['neurons']
            total_params += params
            prev_size = lc['neurons']
        total_params += prev_size * output_size + output_size
        
        st.markdown(f"- Total parameters: **{total_params:,}**")
        st.markdown(f"- Input: **{input_size}** fitur")
        st.markdown(f"- Output: **{output_label}**")

with tab3:
    st.header("Parameter Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hyperparameters")
        
        regression_mode = st.session_state.get('regression_mode', False)
        
        # Optimized defaults for regression
        default_epochs = 500 if regression_mode else 200
        default_batch_size = 128 if regression_mode else 64
        default_lr = 0.001 if regression_mode else 0.005
        
        epochs = st.number_input("Epochs", min_value=10, max_value=1000000, value=default_epochs, step=10,
                                help="Coba 500-1000 epochs")
        batch_size = st.number_input("Batch Size", min_value=8, max_value=1024, value=default_batch_size,
                                     help="Coba 128-256 (lebih stable)")
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, 
                                        value=default_lr, format="%.4f",
                                        help="Coba 0.001-0.0005 (lebih kecil)")
        
        loss_function = st.selectbox(
            "Loss Function",
            options=['Mean Squared Error', 'Mean Absolute Error', 'Categorical Crossentropy'],
            index=1,
            help="Coba MSE atau MAE untuk kasus time-series atau regression, Crossentropy untuk classification"
        )
        
        regularization_l2 = 0.0001
    
    with col2:
        st.subheader("Ringkasan Konfigurasi")
        
        config_summary = {
            "Epochs": epochs,
            "Batch Size": batch_size,
            "Learning Rate": learning_rate,
            "Loss Function": loss_function,
            "Hidden Layers": len(st.session_state.get('layer_config', [])),
            "Output Activation": st.session_state.get('output_activation', 'Softmax')
        }
        
        for key, val in config_summary.items():
            st.markdown(f"- **{key}**: {val}")
    
    st.markdown("---")
    
    if st.button("Mulai Training", type="primary"):
        if 'df' not in st.session_state:
            st.error("Silakan muat dataset terlebih dahulu!")
        elif not st.session_state.get('feature_cols'):
            st.error("Silakan pilih kolom fitur!")
        else:
            df = st.session_state['df']
            feature_cols = st.session_state['feature_cols']
            target_col = st.session_state['target_col']
            normalize_method = st.session_state.get('normalize_method', 'minmax')
            n_classes = st.session_state.get('n_classes', 5)
            
            # if built-in dataset, apply minor preprocessing toggles
            if st.session_state.get('data_source', None) == 'Built-in Sensor Dataset':
                # Make a copy to avoid mutating session data unintentionally
                _df_train = df.copy()
                # Save original target values for metrics (pre-normalization)
                if target_col in _df_train.columns:
                    st.session_state['target_original_values'] = _df_train[target_col].values.astype(np.float32)
                
                # Replace df for a training run
                df = _df_train
            
            X = df[feature_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0)
            
            # Simpan normalisasi parameters untuk digunakan saat prediksi
            if normalize_method == 'minmax':
                x_min = X.min(axis=0)
                x_max = X.max(axis=0)
                st.session_state['x_min'] = x_min
                st.session_state['x_max'] = x_max
                X = (X - x_min) / (x_max - x_min + 1e-8)
            elif normalize_method == 'zscore':
                x_mean = X.mean(axis=0)
                x_std = X.std(axis=0)
                st.session_state['x_mean'] = x_mean
                st.session_state['x_std'] = x_std
                X = (X - x_mean) / (x_std + 1e-8)
            
            regression_mode = st.session_state.get('regression_mode', False)
            # Keep a copy of original targets (pre-trained normalization) - already saved above for built-in
            if 'target_original_values' in st.session_state:
                y_original = st.session_state['target_original_values']
            else:
                y_original = df[target_col].values.astype(np.float32)
            y_raw = df[target_col].values.astype(np.float32)
            if regression_mode:
                y_min = float(y_raw.min())
                y_max = float(y_raw.max())
                denom = y_max - y_min if (y_max - y_min) != 0 else 1.0
                Y = ((y_raw - y_min) / denom).reshape(-1, 1)
                bin_edges = None
                st.session_state['target_min'] = y_min
                st.session_state['target_max'] = y_max
            else:
                Y, bin_edges = discretize_target(y_raw, n_classes)
                st.session_state['bin_edges'] = bin_edges
            
            input_size = X.shape[1]
            if regression_mode:
                output_neurons = 1
            else:
                output_neurons = len(np.unique(Y))
            
            st.info(f"Training: {len(X)} sampel, {input_size} fitur, {'1 output' if regression_mode else f'{output_neurons} kelas'}")

            # Train/Test Split (70/30)
            train_ratio = 0.7
            n_samples = len(X)
            n_train = int(n_samples * train_ratio)
            
            indices = np.random.permutation(n_samples)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
            
            X_train = X[train_indices]
            Y_train = Y[train_indices]
            y_raw_train = y_raw[train_indices]
            
            X_test = X[test_indices]
            Y_test = Y[test_indices]
            y_raw_test = y_raw[test_indices]
            
            st.info(f"Train/Test Split: {len(X_train)} train, {len(X_test)} test")
            
            model = NeuralNetwork()
            layer_config = st.session_state['layer_config']
            
            prev_neurons = input_size
            for i, layer_cfg in enumerate(layer_config):
                model.add(Dense(prev_neurons, layer_cfg['neurons'], 
                              learning_rate=learning_rate))
                if layer_cfg['batchnorm']:
                    model.add(BatchNormalization())
                model.add(get_activation_class(layer_cfg['activation'])())
                if layer_cfg['dropout'] > 0:
                    model.add(Dropout(layer_cfg['dropout']))
                prev_neurons = layer_cfg['neurons']
            
            model.add(Dense(prev_neurons, output_neurons, learning_rate=learning_rate))
            
            # For regression: use Sigmoid (output bounded [0, 1])
            # For classification: use output activation from UI
            if regression_mode:
                output_activation_to_use = 'Sigmoid'
                st.session_state['output_activation_used'] = 'Sigmoid (Bounded [0, 1])'
            else:
                output_activation_to_use = st.session_state['output_activation']
                st.session_state['output_activation_used'] = output_activation_to_use
            
            model.add(get_activation_class(output_activation_to_use)())
            
            LossClass = get_loss_class(loss_function)
            if loss_function == 'Categorical Crossentropy':
                model.set_loss(LossClass(regularization_l2=regularization_l2))
            else:
                model.set_loss(LossClass())
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Untuk regression: 3 charts (Loss, MAE, R²), untuk classification: 2 charts (Loss, Accuracy)
            regression_mode = st.session_state.get('regression_mode', False)
            if regression_mode:
                col_chart1, col_chart2, col_chart3 = st.columns(3)
                with col_chart1:
                    loss_placeholder = st.empty()
                with col_chart2:
                    mae_placeholder = st.empty()
                with col_chart3:
                    r2_placeholder = st.empty()
            else:
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    loss_placeholder = st.empty()
                with col_chart2:
                    acc_placeholder = st.empty()
            
            losses = []
            accuracies = []
            r2_scores = []  # Untuk menyimpan R² selama training
            mae_scores = []  # Untuk menyimpan MAE selama training
            
            # Test metrics untuk tracking overfitting
            test_losses = []
            test_r2_scores = []
            test_mae_scores = []
            
            n_batches = max(len(X_train) // batch_size, 1)
            
            for epoch in range(epochs):
                epoch_loss = 0
                
                # Training dengan train data saja
                indices = np.random.permutation(len(X_train))
                X_shuffled = X_train[indices]
                Y_shuffled = Y_train[indices]
                
                for batch in range(n_batches):
                    batch_start = batch * batch_size
                    batch_end = min(batch_start + batch_size, len(X_train))
                    X_batch = X_shuffled[batch_start:batch_end]
                    Y_batch = Y_shuffled[batch_start:batch_end]
                    
                    output = model.forward(X_batch, training=True)
                    batch_loss = model.loss_function.calculate(output, Y_batch)
                    epoch_loss += batch_loss
                    
                    loss_gradient = model.loss_function.backward(output, Y_batch)
                    model.backward(loss_gradient, epoch)
                
                avg_loss = epoch_loss / n_batches
                losses.append(avg_loss)
                
                if epoch % 5 == 0 or epoch == epochs - 1:
                    # Validation pada TEST data (bukan training data!)
                    val_output = model.forward(X_test, training=False)
                    if regression_mode:
                        preds = val_output.flatten()
                        y_min = st.session_state.get('target_min', 0.0)
                        y_max = st.session_state.get('target_max', 1.0)
                        denom = y_max - y_min if (y_max - y_min) != 0 else 1.0
                        preds_orig = preds * denom + y_min
                        y_original = y_raw_test
                        mae = np.mean(np.abs(preds_orig - y_original))
                        r2 = calculate_r2_score(y_original, preds_orig)
                        accuracy = mae  # Gunakan MAE sebagai accuracy metric
                        accuracies.append(accuracy)
                        mae_scores.append(mae)
                        r2_scores.append(r2)
                        test_mae_scores.append(mae)
                        test_r2_scores.append(r2)
                    else:
                        accuracy = calculate_accuracy(Y_test, val_output)
                        accuracies.append(accuracy)
                    
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    if regression_mode:
                        status_text.markdown(f"Epoch {epoch+1}/{epochs} - 🔴 **Loss**: {avg_loss:.6f} - 🟠 **MAE**: {mae:.4f} - 🟣 **R²**: {r2:.4f}")
                    else:
                        status_text.markdown(f"Epoch {epoch+1}/{epochs} - 🔴 **Loss**: {avg_loss:.6f} - 🟣 **Accuracy**: {accuracy:.4f}")
                    
                    # Plot Loss dengan warna merah
                    loss_df = pd.DataFrame({'Loss': losses})
                    loss_placeholder.line_chart(loss_df, color='#E74C3C')

                    # Ensure we have the same number of epochs as accuracy values
                    acc_epochs = list(range(0, len(losses)))[:len(accuracies)]
                    
                    if regression_mode:
                        # Untuk regression, tampilkan MAE dengan warna orange
                        mae_plot = mae_scores[:len(acc_epochs)]
                        mae_df = pd.DataFrame({'MAE': mae_plot})
                        mae_placeholder.line_chart(mae_df, color='#FF9500')
                        
                        # Untuk regression, tampilkan R² dengan warna ungu
                        r2_plot = r2_scores[:len(acc_epochs)]
                        r2_df = pd.DataFrame({'R²': r2_plot})
                        r2_placeholder.line_chart(r2_df, color='#9B59B6')
                    else:
                        # Untuk classification, tampilkan Accuracy dengan warna hijau
                        accuracies_trimmed = accuracies[:len(acc_epochs)]
                        acc_df = pd.DataFrame({'Accuracy': accuracies_trimmed})
                        acc_placeholder.line_chart(acc_df, color='#27AE60')
            
            progress_bar.progress(1.0)
            
            # Final evaluation pada TEST data (bukan training data!)
            final_output = model.forward(X_test, training=False)
            regression_mode = st.session_state.get('regression_mode', False)
            if regression_mode:
                preds = final_output.flatten()
                y_min = st.session_state.get('target_min', 0.0)
                y_max = st.session_state.get('target_max', 1.0)
                denom = y_max - y_min if (y_max - y_min) != 0 else 1.0
                preds_original = preds * denom + y_min
                y_original = y_raw_test
                final_mae = np.mean(np.abs(preds_original - y_original))
                final_r2 = calculate_r2_score(y_original, preds_original)
                final_accuracy = final_mae  # simpan MAE untuk compatibility
            else:
                final_accuracy = calculate_accuracy(Y_test, final_output)
                final_r2 = None
            
            st.session_state['model'] = model
            st.session_state['losses'] = losses
            st.session_state['accuracies'] = accuracies
            st.session_state['final_accuracy'] = final_accuracy
            st.session_state['final_r2'] = final_r2
            st.session_state['r2_scores'] = r2_scores if regression_mode else []
            st.session_state['test_r2_scores'] = test_r2_scores if regression_mode else []
            st.session_state['test_mae_scores'] = test_mae_scores if regression_mode else []
            st.session_state['X'] = X
            st.session_state['Y'] = Y
            st.session_state['X_test'] = X_test
            st.session_state['Y_test'] = Y_test
            # Save original target values for downstream plotting/metrics
            st.session_state['y_raw'] = y_original
            st.session_state['y_raw_test'] = y_raw_test
            st.session_state['normalize_method_used'] = normalize_method
            
            if regression_mode:
                st.success(f"Training selesai!")
            else:
                st.success(f"Training selesai! Final Accuracy: {final_accuracy:.4f}")

with tab4:
    st.header("Hasil Training dan Prediksi")
    
    if 'model' in st.session_state:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.get('regression_mode', False):
                mae_percent = (st.session_state['final_accuracy'] / st.session_state.get('target_max', 1.0)) * 100
                st.metric("Final MAE", f"{mae_percent:.2f}%")
            else:
                st.metric("Final Accuracy", f"{st.session_state['final_accuracy']:.4f}")
        with col2:
            if st.session_state.get('regression_mode', False) and st.session_state.get('final_r2') is not None:
                st.metric("R² Score", f"{st.session_state['final_r2']:.4f}")
            else:
                st.metric("Final Loss", f"{st.session_state['losses'][-1]:.6f}")
        with col3:
            st.metric("Total Epochs", len(st.session_state['losses']))
        
        st.subheader("Training History")
        
        regression_mode = st.session_state.get('regression_mode', False)
        
        if regression_mode:
            # Untuk regression: Loss dan R² Score
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1 = axes[0]
            ax1.plot(st.session_state['losses'], color='#E74C3C', linewidth=1.5)
            ax1.fill_between(range(len(st.session_state['losses'])), 
                            st.session_state['losses'], alpha=0.3, color='#E74C3C')
            ax1.set_xlabel('Epoch', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            ax2 = axes[1]
            # Ensure we have matching lengths for plotting
            acc_epochs = list(range(0, len(st.session_state['losses'])))[:len(st.session_state['accuracies'])]
            r2_scores = st.session_state.get('r2_scores', [])[:len(acc_epochs)]
            
            if r2_scores:
                ax2.plot(acc_epochs, r2_scores, color='#9B59B6', 
                        marker='o', linewidth=2, markersize=5)
                ax2.fill_between(acc_epochs, r2_scores, alpha=0.3, color='#9B59B6')
                ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Perfect (R²=1.0)')
                ax2.set_xlabel('Epoch', fontsize=11)
                ax2.set_ylabel('R² Score', fontsize=11)
                ax2.set_title('Accuracy - R² Growth', fontsize=13, fontweight='bold')
                ax2.set_ylim([-0.1, 1.1])
                ax2.legend(loc='lower right')
            
            ax2.grid(True, alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
        else:
            # Untuk classification: Loss dan Accuracy
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1 = axes[0]
            ax1.plot(st.session_state['losses'], color='#E74C3C', linewidth=1.5)
            ax1.fill_between(range(len(st.session_state['losses'])), 
                            st.session_state['losses'], alpha=0.3, color='#E74C3C')
            ax1.set_xlabel('Epoch', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            ax2 = axes[1]
            acc_epochs = list(range(0, len(st.session_state['losses'])))[:len(st.session_state['accuracies'])]
            accuracies = st.session_state['accuracies'][:len(acc_epochs)]
            ax2.plot(acc_epochs, accuracies, color='#27AE60', 
                    marker='o', linewidth=1.5, markersize=4)
            ax2.fill_between(acc_epochs, accuracies, alpha=0.3, color='#27AE60')
            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('Accuracy', fontsize=11)
            ax2.set_title('Training Accuracy', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Analisis Prediksi")
        
        regression_mode = st.session_state.get('regression_mode', False)
        model_obj = st.session_state['model']
        
        # Gunakan test data untuk analisis (konsisten dengan train/test split)
        X_test = st.session_state.get('X_test', st.session_state.get('X'))
        Y_test = st.session_state.get('Y_test', st.session_state.get('Y'))
        y_raw_test = st.session_state.get('y_raw_test', st.session_state.get('y_raw'))
        
        if regression_mode:
            preds_proba = model_obj.predict_proba(X_test).flatten()
            y_min = st.session_state.get('target_min', None)
            y_max = st.session_state.get('target_max', None)
            if y_min is None or y_max is None:
                preds_orig = preds_proba
            else:
                denom = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
                preds_orig = preds_proba * denom + y_min
            Y_analysis = y_raw_test
        else:
            predictions = model_obj.predict(X_test)
            Y_analysis = Y_test
        
        col1 = st.columns(1)[0]
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            if regression_mode:
                # Plot predicted vs true scatter
                ax.scatter(range(len(Y_analysis)), Y_analysis, label='True', alpha=0.5, s=10)
                ax.scatter(range(len(preds_orig)), preds_orig, label='Predicted', alpha=0.5, s=10)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Soil Moisture')
                ax.set_title('True vs Predicted (Test Set)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                unique_classes = np.unique(Y_analysis)
                pred_counts = [np.sum(predictions == c) for c in unique_classes]
                true_counts = [np.sum(Y_analysis == c) for c in unique_classes]
                x = np.arange(len(unique_classes))
                width = 0.35
                bars1 = ax.bar(x - width/2, true_counts, width, label='Actual', color='#3498DB', alpha=0.8)
                bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', color='#E74C3C', alpha=0.8)
                ax.set_xlabel('Kelas Soil Moisture')
                ax.set_ylabel('Jumlah Sampel')
                ax.set_title('Distribusi Kelas: Actual vs Predicted')
                ax.set_xticks(x)
                ax.set_xticklabels([f'Kelas {c}' for c in unique_classes])
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        
        st.subheader("Evaluasi Detail pada Test Set")
        
        if regression_mode:
            # Untuk regression: gunakan TEST SET langsung (30% data)
            st.markdown("**Prediksi pada Test Set (30% dari seluruh data):**")
            
            # Langsung gunakan test set yang sudah ada
            if 'X_test' in st.session_state and 'y_raw_test' in st.session_state:
                X_test = st.session_state['X_test']
                y_true = st.session_state['y_raw_test']
                
                # Prediksi
                pred_normalized = model_obj.predict_proba(X_test).flatten()
                y_min = st.session_state.get('target_min', 0.0)
                y_max = st.session_state.get('target_max', 1.0)
                denom = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
                pred_orig = pred_normalized * denom + y_min
                
                # Hitung metrics
                abs_errors = np.abs(pred_orig - y_true)
                mae = abs_errors.mean()
                mae_percent = (mae / y_max) * 100  # MAE dalam persen dari range
                r2 = calculate_r2_score(y_true, pred_orig)
                rmse = np.sqrt(np.mean((pred_orig - y_true) ** 2))
                rmse_percent = (rmse / y_max) * 100  # RMSE dalam persen
                mape = np.mean(np.abs((y_true - pred_orig) / (y_true + 1e-8))) * 100  # Mean Absolute Percentage Error
                
                # Tampilkan metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{mae_percent:.2f}%", help=f"Mean Absolute Error: {mae:.0f} dari range {y_max:.0f}")
                with col2:
                    st.metric("RMSE", f"{rmse_percent:.2f}%", help=f"Root Mean Squared Error: {rmse:.0f}")
                with col3:
                    st.metric("R² Score", f"{r2:.4f}", help="Coefficient of Determination (0-1, lebih tinggi lebih baik)")
                with col4:
                    st.metric("Test Samples", f"{len(y_true):,}")
                
                # Sample preview (ambil 300 sampel random untuk ditampilkan)
                sample_size = min(300, len(y_true))
                sample_idx = np.random.choice(len(y_true), sample_size, replace=False)
                
                results_df = pd.DataFrame({
                    'Index': sample_idx,
                    'True Value': y_true[sample_idx],
                    'Predicted': pred_orig[sample_idx],
                    'Error': abs_errors[sample_idx],
                    'Error %': (abs_errors[sample_idx] / (y_true[sample_idx] + 1e-8)) * 100
                })
                
                st.markdown("**Preview 300 Sample dari Test Set:**")
                st.dataframe(
                    results_df.style.format({
                        'True Value': '{:.2f}',
                        'Predicted': '{:.2f}',
                        'Error': '{:.2f}',
                        'Error %': '{:.2f}%'
                    }).background_gradient(subset=['Error %'], cmap='RdYlGn_r', vmin=0, vmax=30),
                    width='stretch'
                )
                
                # Analisis distribusi error
                with st.expander("Analisis Statistik Distribusi Error"):
                    min_error = abs_errors.min() / (y_max - y_min) * 100
                    max_error = abs_errors.max() / (y_max - y_min) * 100
                    median_error = np.median(abs_errors) / (y_max - y_min) * 100
                    std_error = abs_errors.std() / (y_max - y_min) * 100

                    error_stats = {
                        'Min Error': min_error,
                        'Max Error': max_error,
                        'Median Error': median_error,
                        'Std Error': std_error,
                    }
                    for key, val in error_stats.items():
                        st.write(f"- **{key}**: {val:.4f}%")
            
            else:
                df = st.session_state.get('df', None)
                feature_cols = st.session_state.get('feature_cols', [])
                target_col = st.session_state.get('target_col', None)
                
                if df is not None and feature_cols and target_col:
                    # Debug: Tampilkan normalisasi info
                    with st.expander("Debug Info"):
                        st.write("**Normalisasi Parameters:**")
                        st.write(f"- Method: {st.session_state.get('normalize_method_used', 'N/A')}")
                        st.write(f"- Target Min: {st.session_state.get('target_min', 'N/A')}")
                        st.write(f"- Target Max: {st.session_state.get('target_max', 'N/A')}")
                        st.write(f"- Output Activation (UI): {st.session_state.get('output_activation', 'N/A')}")
                        st.write(f"- Output Activation (Used): {st.session_state.get('output_activation_used', 'N/A')} ✅ For regression, bounded to [0, 1]")
                        
                        if st.session_state.get('normalize_method_used') == 'minmax':
                            x_min = st.session_state.get('x_min', None)
                            x_max = st.session_state.get('x_max', None)
                            if x_min is not None:
                                st.write(f"- X Min (per feature): {x_min[:3]}...")
                                st.write(f"- X Max (per feature): {x_max[:3]}...")
                        
                        st.write("\n**Training Data Range:**")
                        st.write(f"- Y normalized range: [0, 1]")
                        st.write(f"- Y original range: [{st.session_state.get('target_min', 'N/A')}, {st.session_state.get('target_max', 'N/A')}]")
                        
                        st.write("\n**Denormalisasi Formula:**")
                        y_min = st.session_state.get('target_min', 0.0)
                        y_max = st.session_state.get('target_max', 1.0)
                        st.write(f"pred_original = pred_normalized * ({y_max} - {y_min}) + {y_min}")
                        st.write(f"pred_original = pred_normalized * {y_max - y_min} + {y_min}")
                    
                    # Ambil 5 sampel random
                    sample_indices = np.random.choice(len(df), min(5, len(df)), replace=False)
                    
                    # Extract fitur dan target dari dataset asli
                    X_sample = df.iloc[sample_indices][feature_cols].values.astype(np.float32)
                    X_sample = np.nan_to_num(X_sample, nan=0.0)
                    
                    # Normalisasi menggunakan parameters dari training data (KONSISTEN)
                    normalize_method_used = st.session_state.get('normalize_method_used', 'minmax')
                    if normalize_method_used == 'minmax':
                        x_min = st.session_state.get('x_min', None)
                        x_max = st.session_state.get('x_max', None)
                        if x_min is not None and x_max is not None:
                            X_sample = (X_sample - x_min) / (x_max - x_min + 1e-8)
                        else:
                            st.error("Normalisasi parameters tidak ditemukan. Lakukan training terlebih dahulu.")
                            st.stop()
                    elif normalize_method_used == 'zscore':
                        x_mean = st.session_state.get('x_mean', None)
                        x_std = st.session_state.get('x_std', None)
                        if x_mean is not None and x_std is not None:
                            X_sample = (X_sample - x_mean) / (x_std + 1e-8)
                        else:
                            st.error("Normalisasi parameters tidak ditemukan. Lakukan training terlebih dahulu.")
                            st.stop()
                    
                    y_sample = df.iloc[sample_indices][target_col].values.astype(np.float32)
                    
                    # Prediksi dengan model
                    pred_sample = model_obj.predict_proba(X_sample).flatten()
                    
                    y_min = st.session_state.get('target_min', 0.0)
                    y_max = st.session_state.get('target_max', 1.0)
                    denom = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
                    pred_sample_orig = pred_sample * denom + y_min
                    
                    # Buat DataFrame hasil
                    results_df = pd.DataFrame({
                        'Sample Index': sample_indices,
                        'True Value': y_sample,
                        'Predicted Value': pred_sample_orig,
                        'Abs Error': np.abs(pred_sample_orig - y_sample)
                    })
                    
                    avg_error = results_df['Abs Error'].mean()
                    r2_test = calculate_r2_score(y_sample, pred_sample_orig)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Abs Error", f"{avg_error:.3f}")
                    with col2:
                        st.metric("R² Score", f"{r2_test:.4f}")
                    
                        st.dataframe(
                            results_df.style.format({
                                'True Value': '{:.2f}',
                                'Predicted Value': '{:.2f}',
                                'Abs Error': '{:.2f}'
                            }),
                            width='stretch'
                        )
                else:
                    st.warning("⚠️ Test set tidak tersedia. Lakukan training terlebih dahulu.")
        else:
            # Untuk classification: tampilkan dari predictions yang sudah ada
            n_samples = len(Y)
            sample_size = min(100, n_samples)
            sample_idx = np.random.choice(n_samples, sample_size, replace=False)
            
            results_df = pd.DataFrame({
                'Sample': sample_idx,
                'True Class': Y[sample_idx],
                'Predicted Class': predictions[sample_idx],
                'Correct': predictions[sample_idx] == Y[sample_idx]
            })
            correct_pct = results_df['Correct'].mean() * 100
            st.markdown(f"**Akurasi Sample**: {correct_pct:.1f}%")
            
            st.dataframe(
                results_df.style.apply(
                    lambda x: ['background-color: #d4edda' if v else 'background-color: #f8d7da' for v in x], subset=['Correct']
                ),
                width='stretch'
            )
        
    else:
        st.info("Lakukan training terlebih dahulu untuk melihat hasil")

st.markdown("---")
st.markdown("Project ini dibuat oleh Group 1")
st.markdown("""
Anggota: 
- G6401231003	Aldi Pramudya
- G6401231027	Raihan Putra Kirana
- G6401231074	Aghnat Hasya Sayyidina
- G6401231102	Rafif Muhammad Farras
- G6401231105	Nugraha Darmaputra Tangkeallo
""")
