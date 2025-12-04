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
from app.data.dataset import generate_soil_moisture_dataset
from app.function.metrics import calculate_accuracy

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
        'ReLU': ReLU,
        'Sigmoid': Sigmoid,
        'Tanh': Tanh,
        'Softmax': Softmax,
        'Linear': Linear
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
    
    if layer_names is None:
        layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(n_layers - 2)] + ['Output']
    
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


DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/soil_moisture.csv')

with st.sidebar:
    st.header("Konfigurasi Data")

    # Pilih sumber dataset (mutually exclusive)
    data_source = st.radio("Sumber dataset", 
                           options=["Built-in (CSV)", "Upload CSV", "Synthetic (Generate)"],
                           index=2)

    df = None

    # Synthetic dataset options
    if data_source == "Synthetic (Generate)":
        st.subheader('Buat dataset sintetis')
        synth_rows = st.number_input("Jumlah baris per lokasi (period days)", min_value=30, max_value=2000, value=365)
        synth_locations = st.number_input("Jumlah lokasi sintetis", min_value=1, max_value=100, value=5)
        synth_seed = st.number_input("Seed (random)", min_value=0, max_value=10000, value=42, step=1)
        if st.button('Generate dataset'):
            df = generate_soil_moisture_dataset(n_rows=synth_rows, seed=int(synth_seed),
                                                save_csv=False, path=DATA_PATH, plot=False,
                                                add_time=True, n_locations=int(synth_locations), period_days=synth_rows)
            st.session_state['df'] = df
            st.session_state['feature_cols'] = [c for c in ['latitude','longitude','clay_content','sand_content','silt_content','sm_aux'] if c in df.columns] or [c for c in ['temperature','humidity','rainfall','cloud_cover'] if c in df.columns]
            st.session_state['target_col'] = 'sm_tgt' if 'sm_tgt' in df.columns else 'soil_moisture'
            st.success(f'Dataset sintetis berhasil dibuat: {len(df)} baris ({synth_locations} lokasi)')

    # Built-in dataset options
    elif data_source == "Built-in (CSV)":
        st.subheader("Dataset Bawaan")
        if os.path.exists(DATA_PATH):
            sample_size = st.slider("Jumlah sampel data (sampling)", 1000, 50000, 10000, step=1000,
                                   help="Dataset asli memiliki 300k+ baris, sampling untuk efisiensi")
            if st.button("Muat Dataset Bawaan"):
                try:
                    df_full = pd.read_csv(DATA_PATH)
                    df = df_full.sample(n=min(sample_size, len(df_full)), random_state=42)
                    st.session_state['df'] = df
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

        default_features = ['latitude', 'longitude', 'clay_content', 'sand_content',
                            'silt_content', 'sm_aux']
        default_features = [f for f in default_features if f in numeric_columns]

        feature_cols = st.multiselect(
            "Kolom Fitur (X)",
            options=numeric_columns,
            default=default_features if default_features else (numeric_columns[:-1] if len(numeric_columns) > 1 else numeric_columns)
        )

        default_target = 'sm_tgt' if 'sm_tgt' in numeric_columns else (numeric_columns[-1] if numeric_columns else None)
        target_col = None
        if numeric_columns:
            target_col = st.selectbox(
                "Kolom Target (Y)",
                options=numeric_columns,
                index=numeric_columns.index(default_target) if default_target in numeric_columns else 0
            )

        n_classes = st.slider("Jumlah kelas (diskritisasi target)", 3, 10, 5,
                             help="Target kontinyu akan di-bin menjadi n kelas")

        st.subheader("Preprocessing")
        normalize_method = st.selectbox(
            "Metode Normalisasi",
            options=['minmax', 'zscore', 'none'],
            format_func=lambda x: {
                'none': 'Tidak Ada',
                'minmax': 'Min-Max Scaling (0-1)',
                'zscore': 'Z-Score Standardization'
            }.get(x, x)
        )

        # Simpan ke session state
        st.session_state['df'] = df
        st.session_state['feature_cols'] = feature_cols
        st.session_state['target_col'] = target_col
        st.session_state['normalize_method'] = normalize_method
        st.session_state['n_classes'] = n_classes

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
            if 'sm_tgt' in df.columns:
                st.metric("Rata-rata SM", f"{df['sm_tgt'].mean():.3f}")
        
        st.subheader("Deskripsi Dataset")
        st.markdown("""
        Dataset ini berisi observasi remote sensing kelembaban tanah untuk tahun 2013 di wilayah Jerman:
        - **time**: Timestamp observasi
        - **latitude/longitude**: Koordinat lokasi
        - **clay_content**: Persentase kandungan tanite
        - **sand_content**: Persentase kandungan pasir
        - **silt_content**: Persentase kandungan lanau
        - **sm_aux**: Kelembaban tanah dari satelit SMOS-ASCAT (smoothed)
        - **sm_tgt**: Kelembaban tanah dari satelit AMSR (target prediksi)
        """)
        
        st.subheader("Sample Data")
        st.dataframe(df.head(15), width='stretch')
        
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe().T.style.format("{:.4f}"), width='stretch')
        
        st.subheader("Visualisasi Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'sm_aux' in df.columns and 'sm_tgt' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(df['sm_aux'], df['sm_tgt'], 
                                    c=df['sm_tgt'], cmap='YlGnBu', 
                                    alpha=0.5, s=10)
                ax.set_xlabel('SM Auxiliary (SMOS-ASCAT)')
                ax.set_ylabel('SM Target (AMSR)')
                ax.set_title('Perbandingan Soil Moisture: SMOS vs AMSR')
                plt.colorbar(scatter, ax=ax, label='SM Target')
                ax.plot([0, 0.6], [0, 0.6], 'r--', alpha=0.7, label='Perfect fit')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        with col2:
            if 'latitude' in df.columns and 'longitude' in df.columns and 'sm_tgt' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(df['longitude'], df['latitude'], 
                                    c=df['sm_tgt'], cmap='Blues', 
                                    alpha=0.6, s=15)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title('Distribusi Spasial Kelembaban Tanah')
                plt.colorbar(scatter, ax=ax, label='Soil Moisture')
                ax.grid(True, alpha=0.3)
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
    - Matplotlib
    - Networkx (untuk visualisasi)
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
                    min_value=4, max_value=256, value=64 if i == 0 else 32,
                    key=f"neurons_{i}"
                )
                activation = st.selectbox(
                    f"Aktivasi",
                    options=['ReLU', 'Sigmoid', 'Tanh', 'Linear'],
                    key=f"activation_{i}"
                )
            
            with col_b:
                use_batchnorm = st.checkbox(f"Batch Normalization", value=True, key=f"bn_{i}")
                dropout_rate = st.slider(f"Dropout", 0.0, 0.5, 0.1, step=0.05, key=f"dropout_{i}")
            
            layer_config.append({
                'neurons': neurons,
                'activation': activation,
                'batchnorm': use_batchnorm,
                'dropout': dropout_rate
            })
            activations_list.append(activation)
        
        st.markdown("**Output Layer**")
        if st.session_state.get('regression_mode', False):
            output_activation = 'Linear'
            st.write('Output Activation: **Linear** (regression mode)')
        else:
            output_activation = st.selectbox(
                "Aktivasi Output",
                options=['ReLU', 'Softmax', 'Sigmoid', 'Linear'],
            )
        activations_list.append(output_activation)
        
        st.session_state['layer_config'] = layer_config
        st.session_state['output_activation'] = output_activation
    
    with col2:
        st.subheader("Visualisasi Arsitektur")
        
        if 'df' in st.session_state and st.session_state.get('feature_cols'):
            input_size = len(st.session_state['feature_cols'])
            output_size = st.session_state.get('n_classes', 5)
        else:
            input_size = 6
            output_size = 5
        
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
        st.markdown(f"- Output: **{output_size}** kelas")

with tab3:
    st.header("Parameter Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hyperparameters")
        
        epochs = st.number_input("Epochs", min_value=10, max_value=5000, value=200, step=10)
        batch_size = st.number_input("Batch Size", min_value=8, max_value=512, value=64)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, 
                                        value=0.005, format="%.4f")
        
        loss_function = st.selectbox(
            "Loss Function",
            options=['Mean Squared Error', 'Categorical Crossentropy', 'Mean Absolute Error']
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
            
            X = df[feature_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0)
            X = normalize_data(X, normalize_method)
            
            regression_mode = st.session_state.get('regression_mode', False)
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
            actual_classes = len(np.unique(Y)) if not regression_mode else 1
            
            st.info(f"Training: {len(X)} sampel, {input_size} fitur, {actual_classes} kelas")
            
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
            
            model.add(Dense(prev_neurons, actual_classes, learning_rate=learning_rate))
            model.add(get_activation_class(st.session_state['output_activation'])())
            
            LossClass = get_loss_class(loss_function)
            if loss_function == 'Categorical Crossentropy':
                model.set_loss(LossClass(regularization_l2=regularization_l2))
            else:
                model.set_loss(LossClass())
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                loss_placeholder = st.empty()
            with col_chart2:
                acc_placeholder = st.empty()
            
            losses = []
            accuracies = []
            
            n_batches = max(len(X) // batch_size, 1)
            
            for epoch in range(epochs):
                epoch_loss = 0
                
                indices = np.random.permutation(len(X))
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]
                
                for batch in range(n_batches):
                    batch_start = batch * batch_size
                    batch_end = min(batch_start + batch_size, len(X))
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
                    val_output = model.forward(X, training=False)
                    if regression_mode:
                        preds = val_output.flatten()
                        y_min = st.session_state.get('target_min', 0.0)
                        y_max = st.session_state.get('target_max', 1.0)
                        denom = y_max - y_min if (y_max - y_min) != 0 else 1.0
                        preds_orig = preds * denom + y_min
                        y_original = y_raw
                        accuracy = np.mean(np.abs(preds_orig - y_original))
                    else:
                        accuracy = calculate_accuracy(Y, val_output)
                    accuracies.append(accuracy)
                    
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    if regression_mode:
                        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - MAE: {accuracy:.4f}")
                    else:
                        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Accuracy: {accuracy:.4f}")
                    
                    loss_df = pd.DataFrame({'Epoch': range(len(losses)), 'Loss': losses})
                    loss_placeholder.line_chart(loss_df.set_index('Epoch'))
                    
                    # Ensure we have the same number of epochs as accuracy values
                    acc_epochs = list(range(0, len(losses)))[:len(accuracies)]
                    # Trim accuracies to match the length of acc_epochs if needed
                    accuracies = accuracies[:len(acc_epochs)]
                    acc_df = pd.DataFrame({'Epoch': acc_epochs, 'Accuracy': accuracies})
                    acc_placeholder.line_chart(acc_df.set_index('Epoch'))
            
            progress_bar.progress(1.0)
            
            final_output = model.forward(X, training=False)
            regression_mode = st.session_state.get('regression_mode', False)
            if regression_mode:
                preds = final_output.flatten()
                y_min = st.session_state.get('target_min', 0.0)
                y_max = st.session_state.get('target_max', 1.0)
                denom = y_max - y_min if (y_max - y_min) != 0 else 1.0
                preds_original = preds * denom + y_min
                y_original = y_raw
                final_accuracy = np.mean(np.abs(preds_original - y_original))
            else:
                final_accuracy = calculate_accuracy(Y, final_output)
            
            st.session_state['model'] = model
            st.session_state['losses'] = losses
            st.session_state['accuracies'] = accuracies
            st.session_state['final_accuracy'] = final_accuracy
            st.session_state['X'] = X
            st.session_state['Y'] = Y
            st.session_state['y_raw'] = y_raw
            
            st.success(f"Training selesai! Final Accuracy: {final_accuracy:.4f}")

with tab4:
    st.header("Hasil Training dan Prediksi")
    
    if 'model' in st.session_state:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.get('regression_mode', False):
                st.metric("Final MAE", f"{st.session_state['final_accuracy']:.4f}")
            else:
                st.metric("Final Accuracy", f"{st.session_state['final_accuracy']:.4f}")
        with col2:
            st.metric("Final Loss", f"{st.session_state['losses'][-1]:.6f}")
        with col3:
            st.metric("Total Epochs", len(st.session_state['losses']))
        
        st.subheader("Training History")
        
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
        X = st.session_state['X']
        Y = st.session_state['Y'] if not regression_mode else st.session_state['y_raw']
        if regression_mode:
            preds_proba = model_obj.predict_proba(X).flatten()
            y_min = st.session_state.get('target_min', None)
            y_max = st.session_state.get('target_max', None)
            if y_min is None or y_max is None:
                preds_orig = preds_proba
            else:
                denom = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
                preds_orig = preds_proba * denom + y_min
        else:
            predictions = model_obj.predict(st.session_state['X'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            if regression_mode:
                # Plot predicted vs true scatter
                ax.scatter(range(len(Y)), Y, label='True', alpha=0.5, s=10)
                ax.scatter(range(len(preds_orig)), preds_orig, label='Predicted', alpha=0.5, s=10)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Soil Moisture')
                ax.set_title('True vs Predicted (Regression)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                unique_classes = np.unique(Y)
                pred_counts = [np.sum(predictions == c) for c in unique_classes]
                true_counts = [np.sum(Y == c) for c in unique_classes]
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
        
        with col2:
            if regression_mode:
                # Plot residuals distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                residuals = preds_orig - Y
                ax.hist(residuals, bins=30, color='orange', edgecolor='white', alpha=0.7)
                ax.set_xlabel('Residuals (Predicted - True)')
                ax.set_title('Residuals Distribution')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                if 'y_raw' in st.session_state and 'bin_edges' in st.session_state:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    correct = predictions == Y
                    y_raw = st.session_state['y_raw']
                    ax.scatter(range(len(y_raw)), y_raw, c=correct, cmap='RdYlGn', alpha=0.5, s=10)
                    ax.set_xlabel('Sample Index')
                    ax.set_ylabel('Soil Moisture (sm_tgt)')
                    ax.set_title('Prediksi per Sampel (Hijau=Benar, Merah=Salah)')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        st.subheader("Sample Prediksi")
        
        n_samples = len(Y) if not regression_mode else len(Y)
        sample_size = min(100, n_samples)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        
        if regression_mode:
            results_df = pd.DataFrame({
                'Sample': sample_idx,
                'True Value': Y[sample_idx],
                'Predicted Value': preds_orig[sample_idx],
                'Abs Error': np.abs(preds_orig[sample_idx] - Y[sample_idx])
            })
            avg_error = results_df['Abs Error'].mean()
            st.markdown(f"**Avg Abs Error (sample)**: {avg_error:.3f}")
        else:
            results_df = pd.DataFrame({
                'Sample': sample_idx,
                'True Class': Y[sample_idx],
                'Predicted Class': predictions[sample_idx],
                'Correct': predictions[sample_idx] == Y[sample_idx]
            })
            correct_pct = results_df['Correct'].mean() * 100
            st.markdown(f"**Akurasi Sample**: {correct_pct:.1f}%")
        
            if regression_mode:
                st.dataframe(results_df.style.format({'True Value':'{:.3f}','Predicted Value':'{:.3f}','Abs Error':'{:.3f}'}), width='stretch')
            else:
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
