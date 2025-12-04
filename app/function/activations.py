import numpy as np

class BaseActivation:
    # Class ini adalah class dasar untuk semua activation function
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, dvalues):
        raise NotImplementedError

class ReLU(BaseActivation):
    # Activation function ReLU
    def forward(self, inputs):
        """Melakukan proses forward untuk activation function ReLU"""
        self.inputs = inputs
        self.output = np.maximum(0, inputs)  # Mengembalikan nilai terbesar antara 0 dan input
        return self.output

    def backward(self, dvalues):
        """Melakukan proses backward untuk activation function ReLU"""
        self.dinputs = dvalues.copy()  # Mengcopy nilai dvalues
        self.dinputs[self.inputs <= 0] = 0  # Mengatur nilai dvalues menjadi 0 di tempat input <= 0
        return self.dinputs

class Tanh(BaseActivation):
    # Activation function Tanh
    def forward(self, inputs):
        """Melakukan proses forward untuk activation function Tanh"""
        self.inputs = inputs
        self.output = np.tanh(inputs)  # Menghitung tangens hiperbolik
        return self.output
        
    def backward(self, dvalues):
        """Melakukan proses backward untuk activation function Tanh"""
        self.dinputs = dvalues * (1 - np.square(self.output))  # Menghitung gradient input
        return self.dinputs

class Sigmoid(BaseActivation):
    # Activation function Sigmoid
    def forward(self, inputs):
        """Melakukan proses forward untuk activation function Sigmoid"""
        self.inputs = inputs
        # Menghitung nilai sigmoid dengan mencapai batasan input
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500))) 
        return self.output
        
    def backward(self, dvalues):
        """Melakukan proses backward untuk activation function Sigmoid"""
        self.dinputs = dvalues * self.output * (1 - self.output)  # Menghitung gradient input
        return self.dinputs

class Softmax(BaseActivation):
    # Activation function Softmax
    def forward(self, inputs):
        """Melakukan proses forward untuk activation function Softmax"""
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # Menghitung nilai eksponen
        # Menghitung nilai probabilitas dengan normalisasi
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) 
        self.output = probabilities  # Mengembalikan nilai probabilitas
        return self.output

    def backward(self, dvalues):
        """Melakukan proses backward untuk activation function Softmax"""
        batch_size = len(dvalues)        
        self.dinputs = np.zeros_like(dvalues)  # Membuat array dinputs dengan ukuran sama dengan dvalues
        
        for i in range(batch_size):
            output_single = self.output[i].reshape(-1, 1)  # Mengubah output menjadi array 2D
            jacobian_matrix = output_single * (np.eye(len(output_single)) - output_single.T)  # Menghitung jacobian matrix
            self.dinputs[i] = np.dot(jacobian_matrix, dvalues[i])  # Menghitung gradient input
        
        return self.dinputs

class Linear(BaseActivation):
    # Activation function Linear (untuk regresi)
    def forward(self, inputs):
        """Melakukan proses forward untuk activation function Linear"""
        self.inputs = inputs
        self.output = inputs  # Output sama dengan input
        return self.output

    def backward(self, dvalues):
        """Melakukan proses backward untuk activation function Linear"""
        self.dinputs = dvalues.copy()  # Gradient input sama dengan dvalues
        return self.dinputs