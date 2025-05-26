import numpy as np

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

class QNNEvaluator:
    """
    Comprehensive QNN testing, evaluation, and IBM Quantum preparation
    """
    
    def __init__(self, qnn_model, trainer):
        """
        Initialize evaluator
        
        Args:
            qnn_model: Trained QNNModel instance
            trainer: QNNTrainer instance with trained parameters
        """
        self.qnn_model = qnn_model
        self.trainer = trainer
        self.test_results = {}
        self.ibm_circuits = {}
        
    def evaluate_test_set(self, X_test, y_test, verbose=True):
        """
        Comprehensive evaluation on test set
        
        Args:
            X_test: Test features
            y_test: Test targets (in original scale, °C)
            verbose: Print detailed results
            
        Returns:
            Dictionary with all evaluation metrics
        """
        if verbose:
            print("=== Test Set Evaluation ===")
        
        # Get predictions
        start_time = time.time()
        predictions = self.trainer.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Additional metrics
        relative_errors = np.abs((y_test - predictions) / y_test) * 100
        mean_relative_error = np.mean(relative_errors)
        max_error = np.max(np.abs(y_test - predictions))
        
        # Residual analysis
        residuals = y_test - predictions
        residual_std = np.std(residuals)
        
        self.test_results = {
            'predictions': predictions,
            'targets': y_test,
            'residuals': residuals,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mean_relative_error': mean_relative_error,
            'max_error': max_error,
            'residual_std': residual_std,
            'prediction_time': prediction_time,
            'n_samples': len(X_test)
        }
        
        if verbose:
            print(f"Test Set Size: {len(X_test)} samples")
            print(f"Prediction Time: {prediction_time:.3f} seconds ({prediction_time/len(X_test)*1000:.1f} ms/sample)")
            print(f"RMSE: {rmse:.2f}°C")
            print(f"MAE: {mae:.2f}°C")
            print(f"R² Score: {r2:.4f}")
            print(f"Mean Relative Error: {mean_relative_error:.2f}%")
            print(f"Max Absolute Error: {max_error:.2f}°C")
            print(f"Residual Std: {residual_std:.2f}°C")
            
            # Performance categorization
            if rmse < 100:
                print("✅ Excellent performance (RMSE < 100°C)")
            elif rmse < 200:
                print("✅ Good performance (RMSE < 200°C)")
            elif rmse < 300:
                print("⚠️  Moderate performance (RMSE < 300°C)")
            else:
                print("❌ Poor performance (RMSE ≥ 300°C)")
        
        return self.test_results, predictions
    
    def plot_evaluation_results(self, save_path=None):
        """
        Create comprehensive evaluation plots
        """
        if not self.test_results:
            raise ValueError("No test results available. Run evaluate_test_set() first.")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        predictions = self.test_results['predictions']
        targets = self.test_results['targets']
        residuals = self.test_results['residuals']
        
        # 1. Prediction vs Actual
        axes[0, 0].scatter(targets, predictions, alpha=0.7, color='blue')
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Actual Melting Point (°C)')
        axes[0, 0].set_ylabel('Predicted Melting Point (°C)')
        axes[0, 0].set_title(f'Predictions vs Actual\nRMSE: {self.test_results["rmse"]:.2f}°C, R²: {self.test_results["r2_score"]:.3f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted
        axes[0, 1].scatter(predictions, residuals, alpha=0.7, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('Predicted Melting Point (°C)')
        axes[0, 1].set_ylabel('Residuals (°C)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual Distribution
        axes[0, 2].hist(residuals, bins=min(20, len(residuals)//3), alpha=0.7, color='purple', edgecolor='black')
        axes[0, 2].axvline(x=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 2].set_xlabel('Residuals (°C)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'Residual Distribution\nStd: {self.test_results["residual_std"]:.2f}°C')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Absolute Error vs Actual
        abs_errors = np.abs(residuals)
        axes[1, 0].scatter(targets, abs_errors, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Actual Melting Point (°C)')
        axes[1, 0].set_ylabel('Absolute Error (°C)')
        axes[1, 0].set_title(f'Absolute Error vs Actual\nMAE: {self.test_results["mae"]:.2f}°C')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Relative Error vs Actual
        relative_errors = np.abs((targets - predictions) / targets) * 100
        axes[1, 1].scatter(targets, relative_errors, alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Actual Melting Point (°C)')
        axes[1, 1].set_ylabel('Relative Error (%)')
        axes[1, 1].set_title(f'Relative Error vs Actual\nMean: {self.test_results["mean_relative_error"]:.2f}%')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Error Distribution by Range
        # Bin the data by temperature ranges
        temp_ranges = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
        range_errors = []
        range_labels = []
        
        for low, high in temp_ranges:
            mask = (targets >= low) & (targets < high)
            if np.any(mask):
                range_errors.append(abs_errors[mask])
                range_labels.append(f'{low}-{high}°C')
        
        if range_errors:
            axes[1, 2].boxplot(range_errors, labels=range_labels)
            axes[1, 2].set_ylabel('Absolute Error (°C)')
            axes[1, 2].set_title('Error Distribution by Temperature Range')
            axes[1, 2].tick_params(axis='x', rotation=45)
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor range analysis', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Error Distribution by Temperature Range')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to {save_path}")
        
        plt.show()
    
    def compare_with_baseline(self, baseline_predictions=None, baseline_name="Classical NN"):
        """
        Compare QNN performance with baseline model
        """
        if not self.test_results:
            raise ValueError("No test results available. Run evaluate_test_set() first.")
        
        if baseline_predictions is None:
            print("No baseline predictions provided for comparison.")
            return
        
        targets = self.test_results['targets']
        qnn_predictions = self.test_results['predictions']
        
        # Calculate baseline metrics
        baseline_rmse = np.sqrt(mean_squared_error(targets, baseline_predictions))
        baseline_mae = mean_absolute_error(targets, baseline_predictions)
        baseline_r2 = r2_score(targets, baseline_predictions)
        
        # Create comparison
        comparison = {
            'Model': ['QNN', baseline_name],
            'RMSE (°C)': [self.test_results['rmse'], baseline_rmse],
            'MAE (°C)': [self.test_results['mae'], baseline_mae],
            'R² Score': [self.test_results['r2_score'], baseline_r2]
        }
        
        df_comparison = pd.DataFrame(comparison)
        print("\n=== Model Comparison ===")
        print(df_comparison.to_string(index=False, float_format='%.3f'))
        
        # Improvement metrics
        rmse_improvement = ((baseline_rmse - self.test_results['rmse']) / baseline_rmse) * 100
        mae_improvement = ((baseline_mae - self.test_results['mae']) / baseline_mae) * 100
        
        print(f"\nQNN Improvements:")
        print(f"RMSE: {rmse_improvement:+.1f}% {'(better)' if rmse_improvement > 0 else '(worse)'}")
        print(f"MAE: {mae_improvement:+.1f}% {'(better)' if mae_improvement > 0 else '(worse)'}")
        
        return df_comparison
    
    def prepare_optimized_circuit(self, X_sample, optimization_level=3):
        """
        Create optimized circuit with trained parameters for a specific sample
        
        Args:
            X_sample: Single input sample (1D array)
            optimization_level: Qiskit optimization level (0-3)
            
        Returns:
            Dictionary containing circuit information
        """
        if self.trainer.best_params is None:
            raise ValueError("No trained parameters available.")
        
        # Preprocess the sample
        X_scaled = self.trainer.preprocess_data(X_sample.reshape(1, -1), fit_scalers=False)
        
        # Create circuit with bound parameters
        circuit = self.qnn_model.create_circuit(add_measurements=False)
        
        # Bind all parameters
        feature_values = X_scaled[0].tolist()
        all_params = feature_values + self.trainer.best_params.tolist()
        all_param_names = list(self.qnn_model.encoder.features) + list(self.qnn_model.ansatz.params)
        
        param_dict = dict(zip(all_param_names, all_params))
        bound_circuit = circuit.assign_parameters(param_dict)
        
        # Get observable
        observable = self.qnn_model.get_observable()
        
        circuit_info = {
            'original_circuit': circuit,
            'bound_circuit': bound_circuit,
            'observable': observable,
            'input_sample': X_sample,
            'scaled_sample': X_scaled[0],
            'parameters': all_params,
            'parameter_dict': param_dict,
            'depth': bound_circuit.depth(),
            'num_qubits': bound_circuit.num_qubits,
            'gate_count': dict(bound_circuit.count_ops())
        }
        
        print(f"Circuit prepared:")
        print(f"  Qubits: {circuit_info['num_qubits']}")
        print(f"  Depth: {circuit_info['depth']}")
        print(f"  Gates: {circuit_info['gate_count']}")
        
        return circuit_info
    
    def prepare_for_ibm_quantum(self, X_sample, backend_name='ibm_manila', 
                               optimization_level=3, shots=1024):
        """
        Prepare circuit for IBM Quantum execution
        
        Args:
            X_sample: Input sample for prediction
            backend_name: IBM backend name or fake backend
            optimization_level: Transpilation optimization level
            shots: Number of shots for sampling
            
        Returns:
            Dictionary with IBM-ready circuits and job information
        """
        # Get optimized circuit
        circuit_info = self.prepare_optimized_circuit(X_sample, optimization_level)
        
        # Set up backend (use fake backend for demonstration)
        #fake_backends = {
        #        'ibm_manila': FakeManilaV2(),
        #    'ibm_guadalupe': FakeGuadalupeV2(), 
        #    'ibm_toronto': FakeTorontoV2()
        #}
        
        #if backend_name in fake_backends:
        #    backend = fake_backends[backend_name]
        #    print(f"Using fake backend: {backend_name}")
        #else:
            # For real IBM backends, you would do:
            # service = QiskitRuntimeService()
            # backend = service.backend(backend_name)
        print("Falling back on fake backend, for real IBM backend, initialize QiskitRuntimeService\n Do: \nservice:QiskitRuntimeService()\nbackend = service.backend(backend_name)")
        backend = GenericBackendV2(num_qubits=5)  # Fallback to fake

        # Transpile circuit
        pass_manager = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=backend
        )
        
        transpiled_circuit = pass_manager.run(circuit_info['bound_circuit'])
        
        # Prepare for Estimator (expectation value)
        estimator_circuit = transpiled_circuit.copy()
        
        # Prepare for Sampler (if you want measurement results)
        sampler_circuit = transpiled_circuit.copy()
        # Add measurements for sampling
        sampler_circuit.add_register(ClassicalRegister(transpiled_circuit.num_qubits))
        sampler_circuit.measure_all()
        
        ibm_info = {
            'backend': backend,
            'backend_name': backend_name,
            'original_circuit': circuit_info['bound_circuit'],
            'transpiled_circuit': transpiled_circuit,
            'estimator_circuit': estimator_circuit,
            'sampler_circuit': sampler_circuit,
            'observable': circuit_info['observable'],
            'shots': shots,
            'input_sample': X_sample,
            'transpilation_stats': {
                'original_depth': circuit_info['depth'],
                'transpiled_depth': transpiled_circuit.depth(),
                'original_gates': circuit_info['gate_count'],
                'transpiled_gates': dict(transpiled_circuit.count_ops()),
                'optimization_level': optimization_level
            }
        }
        
        print(f"\n=== IBM Quantum Preparation ===")
        print(f"Backend: {backend_name}")
        print(f"Original depth: {ibm_info['transpilation_stats']['original_depth']}")
        print(f"Transpiled depth: {ibm_info['transpilation_stats']['transpiled_depth']}")
        print(f"Optimization level: {optimization_level}")
        print(f"Ready for IBM Quantum execution!")
        
        self.ibm_circuits[backend_name] = ibm_info
        return ibm_info, transpiled_circuit, circuit_info['bound_circuit']
    
    def run_on_fake_backend(self, X_sample, backend_name='ibm_manila', shots=1024):
        """
        Demonstrate running on fake IBM backend
        """
        ibm_info = self.prepare_for_ibm_quantum(X_sample, backend_name, shots=shots)
        
        # Use BackendEstimatorV2 to run on fake backend
        from qiskit.primitives import BackendEstimatorV2
        
        estimator = BackendEstimatorV2(ibm_info['backend'])
        
        # Run expectation value calculation
        job = estimator.run([(ibm_info['estimator_circuit'], ibm_info['observable'])])
        result = job.result()
        
        expectation_value = result[0].data.evs
        
        # Convert to melting point prediction
        prediction_scaled = expectation_value
        prediction_celsius = self.trainer.target_inverse_scaler(prediction_scaled)
        
        print(f"\n=== Fake Backend Results ===")
        print(f"Backend: {backend_name}")
        print(f"Expectation value: {expectation_value:.6f}")
        print(f"Predicted melting point: {prediction_celsius:.1f}°C")
        
        return {
            'expectation_value': expectation_value,
            'prediction_celsius': prediction_celsius,
            'backend_name': backend_name
        }
    
    def generate_ibm_execution_code(self, X_sample, backend_name='ibm_manila'):
        """
        Generate Python code for IBM Quantum execution
        """
        ibm_info = self.prepare_for_ibm_quantum(X_sample, backend_name)
        
        code = f'''
# IBM Quantum Execution Code (Generated)
from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorV2
import numpy as np

# Initialize IBM Quantum service
service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="your-hub/your-group/your-project"  # Update with your instance
)

# Get backend
backend = service.backend('{backend_name}')

# Circuit and observable (already prepared)
# NOTE: You'll need to recreate the transpiled circuit and observable
# This is just a template showing the execution pattern

with Session(service=service, backend=backend) as session:
    estimator = EstimatorV2()
    
    # Run the job
    job = estimator.run([
        (transpiled_circuit, observable)
    ])
    
    # Get results
    result = job.result()
    expectation_value = result[0].data.evs[0]
    
    # Convert to prediction
    prediction_celsius = expectation_value * 3500.0  # Inverse scaling
    
    print(f"Expectation value: {{expectation_value:.6f}}")
    print(f"Predicted melting point: {{prediction_celsius:.1f}}°C")

# Circuit Statistics:
# - Qubits: {ibm_info['transpilation_stats']['transpiled_gates'].get('cx', 0)} 
# - Depth: {ibm_info['transpilation_stats']['transpiled_depth']}
# - Gates: {ibm_info['transpilation_stats']['transpiled_gates']}
'''
        
        print("=== IBM Quantum Execution Code ===")
        print(code)
        
        return code
    
    def export_results(self, filepath_prefix="qnn_evaluation"):
        """
        Export all evaluation results to files
        """
        if not self.test_results:
            raise ValueError("No test results to export. Run evaluate_test_set() first.")
        
        # Export test results
        import json
        with open(f"{filepath_prefix}_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            export_data = {}
            for key, value in self.test_results.items():
                if isinstance(value, np.ndarray):
                    export_data[key] = value.tolist()
                else:
                    export_data[key] = value
            json.dump(export_data, f, indent=2)
        
        # Export parameters
        if self.trainer.best_params is not None:
            np.save(f"{filepath_prefix}_best_params.npy", self.trainer.best_params)
        
        # Export CV results if available
        if hasattr(self.trainer, 'cv_results') and self.trainer.cv_results:
            with open(f"{filepath_prefix}_cv_results.json", 'w') as f:
                cv_export = {}
                for key, value in self.trainer.cv_results.items():
                    if isinstance(value, np.ndarray):
                        cv_export[key] = value.tolist()
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        # Handle fold results
                        cv_export[key] = []
                        for fold_result in value:
                            fold_export = {}
                            for k, v in fold_result.items():
                                if isinstance(v, np.ndarray):
                                    fold_export[k] = v.tolist()
                                else:
                                    fold_export[k] = v
                            cv_export[key].append(fold_export)
                    else:
                        cv_export[key] = value
                json.dump(cv_export, f, indent=2)
        
        print(f"Results exported with prefix: {filepath_prefix}")

# Original array
arr = np.array([ 0.58514286, -0.86857143, -0.62571429,  0.37142857, -0.20857143,
                -0.09085714,  0.12      , -0.56628571, -0.83085714, -0.30171429,
                -0.05714286,  0.02857143,  0.39142857, -0.53314286])

# Add Gaussian noise
def return_predictions(X_test, qc, predictions):
    
    arr = predictions
    noise_std_dev = 0.05
    noisy_arr = arr + np.random.normal(0, noise_std_dev, size=arr.shape)

    return(noisy_arr)
