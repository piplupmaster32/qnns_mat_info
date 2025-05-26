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
