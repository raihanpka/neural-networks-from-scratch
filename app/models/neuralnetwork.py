import numpy as np
from ..function.layer import Dense
from ..function.activations import ReLU, Softmax
from ..function.regularization import BatchNormalization, Dropout
from ..function.check_loss import CategoricalCrossentropy, MeanSquaredError, MeanAbsoluteError
from ..function.metrics import calculate_accuracy

class NeuralNetwork:
    def __init__(self):
        """Inisialisasi objek NeuralNetwork"""
        self.layers = []  # List untuk menyimpan layer
        self.loss_function = None  # Loss function yang akan digunakan
        self.history = {'loss': [], 'accuracy': []}  # History loss dan accuracy
    
    def add(self, layer):
        """Tambahkan layer ke dalam NeuralNetwork"""
        self.layers.append(layer)
    
    def set_loss(self, loss_function):
        """Set loss function yang akan digunakan"""
        self.loss_function = loss_function
    
    def forward(self, X, training=True):
        """Lakukan forward pass melalui semua layer"""
        output = X
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                if 'training' in layer.forward.__code__.co_varnames:
                    output = layer.forward(output, training=training)
                else:
                    output = layer.forward(output)
        return output
    
    def backward(self, dvalues, epoch=0):
        """Lakukan backward pass melalui semua layer"""
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                if 'epoch' in layer.backward.__code__.co_varnames:
                    dvalues = layer.backward(dvalues, epoch)
                else:
                    dvalues = layer.backward(dvalues)
        return dvalues
    
    def train(self, X, Y, epochs=100, batch_size=32, patience=30, verbose=True):
        """Latih NeuralNetwork"""
        # Mengatur batch_size jika tidak diberikan
        if batch_size <= 0:
            batch_size = len(X)
        
        n_batches = max(len(X) // batch_size, 1)
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0
            
            # Mengacak data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            for batch in range(n_batches):
                batch_start = batch * batch_size
                batch_end = min(batch_start + batch_size, len(X))
                X_batch = X_shuffled[batch_start:batch_end]
                Y_batch = Y_shuffled[batch_start:batch_end]
                
                # Forward pass
                output = self.forward(X_batch, training=True)
                
                # Menghitung loss
                batch_loss = self.loss_function.calculate(output, Y_batch)
                epoch_loss += batch_loss
                
                # Backward pass
                loss_gradient = self.loss_function.backward(output, Y_batch)
                self.backward(loss_gradient, epoch)
            
            avg_epoch_loss = epoch_loss / n_batches
            
            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch > 100:
                    if verbose:
                        print(f"Pelepasan cepat pada epoch {epoch}")
                    break
            
            # Validasi dan logging
            if epoch % 20 == 0 and verbose:
                val_output = self.forward(X, training=False)
                # If val_output is 1D or has single output, treat as regression and compute MAE
                if val_output.ndim == 1 or (val_output.ndim > 1 and val_output.shape[1] == 1):
                    y_true_vals = Y.flatten() if hasattr(Y, 'flatten') else Y
                    preds_vals = val_output.flatten()
                    accuracy = np.mean(np.abs(preds_vals - y_true_vals))
                    print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.6f}, MAE: {accuracy:.6f}")
                else:
                    accuracy = calculate_accuracy(Y, val_output)
                    print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.6f}, Accuracy: {accuracy:.4f}")

                self.history['loss'].append(avg_epoch_loss)
                self.history['accuracy'].append(accuracy)
    
    def predict(self, X):
        """Prediksi hasil dengan menggunakan output dari model"""
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """Mendapatkan probabilities dari output model"""
        return self.forward(X, training=False)