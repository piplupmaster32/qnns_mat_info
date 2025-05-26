import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from typing import List, Tuple, Dict, Optional, Callable
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from qiskit import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator as RuntimeEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time

class QNNTrainer:
    """
    QNN Training and Optimization Module
    Based on the materials informatics paper methodology
    """
    
    def __init__(self, qnn_model: QNN, optimizer='Powell', learning_rate=0.01, 
                 max_iterations=1000, convergence_threshold=1e-6):
        """
        Initialize QNN Trainer
        
        Args:
            qnn_model: QNNModel instance
            optimizer: Optimization method ('Powell', 'COBYLA', 'L-BFGS-B', 'gradient_descent')
            learning_rate: Learning rate for gradient-based optimizers
            max_iterations: Maximum number of optimization iterations
            convergence_threshold: Convergence criterion
        """
        self.qnn_model = qnn_model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'parameters': [],
            'iteration': []
        }
        
        # Data preprocessing
        self.feature_scaler = None
        self.target_scaler = None
        self.fitted_scalers = False
        
        # Cross-validation results
        self.cv_results = {}
        
        # Current best parameters
        self.best_params = None
        self.best_loss = np.inf
        
    def preprocess_data(self, X, y=None, fit_scalers=True):
        """
        Preprocess data according to paper methodology:
        - Normalize features to mean=0, std=1
        - Scale features to [-1, 1] range
        - Scale target by dividing by 3500 (max melting point scaling)
        """
        if fit_scalers:
            # Initialize scalers
            self.feature_scaler = StandardScaler()
            
            # Fit and transform features
            X_normalized = self.feature_scaler.fit_transform(X)
            
            # Additional scaling to [-1, 1] range
            self.feature_minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
            X_scaled = self.feature_minmax_scaler.fit_transform(X_normalized)
            
            if y is not None:
                # Scale target by 3500 (approximate max melting point)
                self.target_scaler = lambda y: y / 3500.0
                self.target_inverse_scaler = lambda y_scaled: y_scaled * 3500.0
                y_scaled = self.target_scaler(y)
                self.fitted_scalers = True
                return X_scaled, y_scaled
            else:
                self.fitted_scalers = True
                return X_scaled
        else:
            if not self.fitted_scalers:
                raise ValueError("Scalers not fitted. Set fit_scalers=True first.")
            
            # Transform using fitted scalers
            X_normalized = self.feature_scaler.transform(X)
            X_scaled = self.feature_minmax_scaler.transform(X_normalized)
            
            if y is not None:
                y_scaled = self.target_scaler(y)
                return X_scaled, y_scaled
            else:
                return X_scaled
    
    def cost_function(self, params, X, y, return_predictions=False):
        """
        Mean Squared Error cost function
        
        Args:
            params: QNN parameters (features + trainable parameters)
            X: Input features
            y: Target values
            return_predictions: If True, return predictions along with cost
        """
        predictions = []
        
        for i in range(len(X)):
            # Combine feature values with trainable parameters
            feature_values = X[i].tolist()
            all_params = feature_values + params.tolist()
            
            # Get expectation value
            result = self.qnn_model.run_expectation(all_params)
            prediction = result[0].data.evs  # Extract expectation value
            predictions.append(prediction)
        
        predictions = np.array(predictions)
        mse = mean_squared_error(y, predictions)
        
        if return_predictions:
            return mse, predictions
        return mse
    
    def gradient_finite_diff(self, params, X, y, epsilon=1e-8):
        """
        Compute gradients using finite differences
        """
        gradients = np.zeros_like(params)
        f0 = self.cost_function(params, X, y)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            f_plus = self.cost_function(params_plus, X, y)
            gradients[i] = (f_plus - f0) / epsilon
            
        return gradients
    
    def parameter_shift_gradient(self, params, X, y):
        """
        Compute gradients using parameter-shift rule
        More accurate for quantum circuits but slower
        """
        gradients = np.zeros_like(params)
        shift = np.pi / 2  # Standard parameter shift
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += shift
            params_minus[i] -= shift
            
            f_plus = self.cost_function(params_plus, X, y)
            f_minus = self.cost_function(params_minus, X, y)
            
            gradients[i] = (f_plus - f_minus) / 2
            
        return gradients
    
    def train_single_fold(self, X_train, y_train, X_val=None, y_val=None, 
                         initial_params=None, verbose=True):
        """
        Train QNN on a single fold of data
        """
        # Initialize parameters if not provided
        if initial_params is None:
            n_trainable = len(self.qnn_model.get_trainable_parameters())
            initial_params = np.random.uniform(-np.pi, np.pi, n_trainable)
        
        # Clear training history
        self.training_history = {'train_loss': [], 'val_loss': [], 'parameters': [], 'iteration': []}
        
        def callback(params):
            """Callback function to track training progress"""
            train_loss = self.cost_function(params, X_train, y_train)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['parameters'].append(params.copy())
            self.training_history['iteration'].append(len(self.training_history['train_loss']))
            
            if X_val is not None and y_val is not None:
                val_loss = self.cost_function(params, X_val, y_val)
                self.training_history['val_loss'].append(val_loss)
            
            if verbose and len(self.training_history['train_loss']) % 10 == 0:
                iter_num = len(self.training_history['train_loss'])
                if X_val is not None:
                    print(f"Iteration {iter_num}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                else:
                    print(f"Iteration {iter_num}: Train Loss = {train_loss:.6f}")
            
            # Update best parameters
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.best_params = params.copy()
        
        # Optimization
        if verbose:
            print(f"Starting optimization with {self.optimizer}")
        
        start_time = time.time()
        
        if self.optimizer == 'Powell':
            # Powell method (as used in the paper)
            result = minimize(
                fun=lambda params: self.cost_function(params, X_train, y_train),
                x0=initial_params,
                method='Powell',
                callback=callback,
                options={'maxiter': self.max_iterations, 'ftol': self.convergence_threshold}
            )
        
        elif self.optimizer == 'COBYLA':
            result = minimize(
                fun=lambda params: self.cost_function(params, X_train, y_train),
                x0=initial_params,
                method='COBYLA',
                callback=callback,
                options={'maxiter': self.max_iterations}
            )
        
        elif self.optimizer == 'L-BFGS-B':
            result = minimize(
                fun=lambda params: self.cost_function(params, X_train, y_train),
                x0=initial_params,
                method='L-BFGS-B',
                jac=lambda params: self.gradient_finite_diff(params, X_train, y_train),
                callback=callback,
                options={'maxiter': self.max_iterations, 'ftol': self.convergence_threshold}
            )
        
        elif self.optimizer == 'gradient_descent':
            # Simple gradient descent implementation
            params = initial_params.copy()
            
            for iteration in range(self.max_iterations):
                gradients = self.gradient_finite_diff(params, X_train, y_train)
                params -= self.learning_rate * gradients
                
                callback(params)
                
                # Check convergence
                if len(self.training_history['train_loss']) > 1:
                    if abs(self.training_history['train_loss'][-1] - 
                          self.training_history['train_loss'][-2]) < self.convergence_threshold:
                        break
            
            result = type('Result', (), {
                'x': params, 
                'fun': self.training_history['train_loss'][-1],
                'success': True
            })()
        
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f"Optimization completed in {training_time:.2f} seconds")
            print(f"Final train loss: {result.fun:.6f}")
            if X_val is not None and y_val is not None:
                final_val_loss = self.cost_function(result.x, X_val, y_val)
                print(f"Final validation loss: {final_val_loss:.6f}")
        
        return result
    
    def k_fold_cross_validation(self, X, y, k=5, random_state=42, verbose=True):
        """
        Perform k-fold cross-validation as described in the paper
        """
        if verbose:
            print(f"Starting {k}-fold cross-validation")
        
        # Preprocess data
        X_scaled, y_scaled = self.preprocess_data(X, y, fit_scalers=True)
        
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
        
        fold_results = []
        train_rmses = []
        val_rmses = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            if verbose:
                print(f"\n--- Fold {fold + 1}/{k} ---")
            
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold, y_val_fold = y_scaled[train_idx], y_scaled[val_idx]
            
            # Train on this fold
            result = self.train_single_fold(
                X_train_fold, y_train_fold, 
                X_val_fold, y_val_fold,
                verbose=verbose
            )
            
            # Calculate RMSE on original scale
            train_loss, train_pred = self.cost_function(
                result.x, X_train_fold, y_train_fold, return_predictions=True
            )
            val_loss, val_pred = self.cost_function(
                result.x, X_val_fold, y_val_fold, return_predictions=True
            )
            
            # Convert back to original scale for RMSE calculation
            train_pred_orig = self.target_inverse_scaler(train_pred)
            val_pred_orig = self.target_inverse_scaler(val_pred)
            y_train_orig = self.target_inverse_scaler(y_train_fold)
            y_val_orig = self.target_inverse_scaler(y_val_fold)
            
            train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred_orig))
            val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))
            
            train_rmses.append(train_rmse)
            val_rmses.append(val_rmse)
            
            fold_results.append({
                'fold': fold + 1,
                'params': result.x,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_predictions': train_pred_orig,
                'val_predictions': val_pred_orig,
                'train_targets': y_train_orig,
                'val_targets': y_val_orig
            })
            
            if verbose:
                print(f"Fold {fold + 1} - Train RMSE: {train_rmse:.2f}°C, Val RMSE: {val_rmse:.2f}°C")
        
        # Calculate average performance
        avg_train_rmse = np.mean(train_rmses)
        avg_val_rmse = np.mean(val_rmses)
        std_val_rmse = np.std(val_rmses)
        
        self.cv_results = {
            'fold_results': fold_results,
            'avg_train_rmse': avg_train_rmse,
            'avg_val_rmse': avg_val_rmse,
            'std_val_rmse': std_val_rmse,
            'train_rmses': train_rmses,
            'val_rmses': val_rmses
        }
        
        if verbose:
            print(f"\n=== Cross-Validation Results ===")
            print(f"Average Train RMSE: {avg_train_rmse:.2f} ± {np.std(train_rmses):.2f}°C")
            print(f"Average Validation RMSE: {avg_val_rmse:.2f} ± {std_val_rmse:.2f}°C")
            print(f"Generalization Gap: {avg_val_rmse - avg_train_rmse:.2f}°C")
        
        return self.cv_results
    
    def predict(self, X, use_best_params=True):
        """
        Make predictions using trained model
        """
        if self.best_params is None and use_best_params:
            raise ValueError("Model not trained. Run training first.")
        
        if not self.fitted_scalers:
            # If scalers have not been fitted then assume the data is already preprocessed.
            # Define identity transformers with a transform() method so that predict can run.
            print("Warning: Data scalers not fitted. Using identity transformers.")
            class IdentityTransformer:
                def transform(self, X):
                    return X
            self.feature_scaler = IdentityTransformer()
            self.feature_minmax_scaler = IdentityTransformer()
            self.target_scaler = lambda y: y
            self.target_inverse_scaler = lambda y: y
            self.fitted_scalers = True
        
        # Preprocess input
        X_scaled = self.preprocess_data(X, fit_scalers=False)
        
        predictions = []
        params = self.best_params if use_best_params else self.training_history['parameters'][-1]
        
        for i in range(len(X_scaled)):
            feature_values = X_scaled[i].tolist()
            all_params = feature_values + params.tolist()
            
            result = self.qnn_model.run_expectation(all_params)
            prediction = result[0].data.evs
            predictions.append(prediction)
        
        predictions = np.array(predictions)
        
        # Convert back to original scale
        predictions_orig = self.target_inverse_scaler(predictions)
        
        return predictions_orig
    
    def plot_training_history(self, fold_idx=None):
        """
        Plot training history
        """
        if not self.training_history['train_loss']:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['iteration'], self.training_history['train_loss'], 
                label='Training Loss', marker='o', markersize=3)
        if self.training_history['val_loss']:
            plt.plot(self.training_history['iteration'], self.training_history['val_loss'], 
                    label='Validation Loss', marker='s', markersize=3)
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        
        # Parameter evolution
        plt.subplot(1, 2, 2)
        params_array = np.array(self.training_history['parameters'])
        for i in range(min(5, params_array.shape[1])):  # Plot first 5 parameters
            plt.plot(self.training_history['iteration'], params_array[:, i], 
                    label=f'θ_{i}', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save trained model and scalers
        """
        model_data = {
            'best_params': self.best_params,
            'feature_scaler': self.feature_scaler,
            'feature_minmax_scaler': self.feature_minmax_scaler,
            'target_scaler': self.target_scaler,
            'target_inverse_scaler': self.target_inverse_scaler,
            'fitted_scalers': self.fitted_scalers,
            'cv_results': self.cv_results,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model and scalers
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_params = model_data['best_params']
        self.feature_scaler = model_data['feature_scaler']
        self.feature_minmax_scaler = model_data['feature_minmax_scaler']
        self.target_scaler = model_data['target_scaler']
        self.target_inverse_scaler = model_data['target_inverse_scaler']
        self.fitted_scalers = model_data['fitted_scalers']
        self.cv_results = model_data['cv_results']
        self.training_history = model_data['training_history']
        
        print(f"Model loaded from {filepath}")