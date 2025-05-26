# Quantum Neural Networks for Predicting Melting Points of Metal Oxides

## Overview
This project implements a Quantum Neural Network (QNN) to predict melting points of metal oxides using material properties as input features. The QNN leverages quantum circuits as a machine learning model, taking advantage of high-dimensional quantum states and built-in regularization through unitarity to mitigate overfitting on small datasets.

## Features
- **Data Processing**
  - Standardization and scaling of input features
  - Train/test split (80/20) for model evaluation
  - Support for both linear and arctan feature encoding

- **QNN Architecture** 
  - Configurable number of qubits (5-10 recommended)
  - Customizable circuit depth
  - Flexible encoding methods:
    - Linear encoding (θ = πx)
    - Arctan encoding (θ = arctan(x) + π/2)
  - Support for redundant feature encoding

- **Model Training**
  - Powell optimization method
  - Training history visualization
  - Support for batch processing
  - Cross-validation capabilities

- **Evaluation**
  - RMSE and R² metrics
  - Prediction vs actual value plots
  - Circuit visualization tools
  - IBM Quantum execution support

## Installation

```bash
# Install required packages
pip install qiskit pennylane torch scikit-learn numpy scipy matplotlib seaborn tqdm
pip install pylatexenc
pip install qiskit[visualization]
pip install pytket pytket-qiskit
```

## Usage 
```bash
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = pd.read_csv("data.csv")
X_raw = data[feature_cols].values
y_raw = data[target_col].values

# Scale features
scaler_X = StandardScaler()
X_norm = scaler_X.fit_transform(X_raw)
X_scaled = X_norm / np.max(np.abs(X_norm), axis=0)

# Scale target (melting points)
y_scaled = (y_raw / 3500.0) * 2 - 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

```

## Training QNN
```bash
# Initialize QNN model
qnn = QNN(n_qubits=5, depth=2, encoding_method='arctan', entangler_type='linear')

# Train model
trainer = QNNTrainer(qnn, optimizer='Powell', max_iterations=100)
results = trainer.train_single_fold(X_train, y_train, verbose=True)

# Make predictions
predictions = trainer.predict(X_test, use_best_params=True)
```

## Running on IBM Hardware
```bash
# Configure IBM Quantum account
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(token=API_TOKEN, channel="ibm_quantum")

# Connect to backend
service = QiskitRuntimeService()
backend = service.least_busy()

# Prepare and run circuit
transpiled_circuit = qiskit.transpile(bound_circuit, backend=backend, optimization_level=3)
```
