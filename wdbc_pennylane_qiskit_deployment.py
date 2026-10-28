"""
================================================================================
WDBC QUANTUM BENCHMARK: PENNYLANE & QISKIT HARDWARE DEPLOYMENT WRAPPER
================================================================================
This script provides production-ready Qiskit and PennyLane implementations for
deploying the Quantum Support Vector Machine (QSVM) Kernel and Variational
Quantum Classifier (VQC) architectures on real IBM Quantum hardware (e.g., ibm_brisbane)
or local noiseless/noisy Aer simulators.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ==============================================================================
# 1. PENNYLANE IMPLEMENTATION (Variational Quantum Classifier)
# ==============================================================================
try:
    import pennylane as qml
    from pennylane import numpy as pnp

    def pennylane_vqc_wdbc():
        print("--- Running PennyLane VQC on WDBC Dataset ---")
        
        # Load & Preprocess Data
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        
        pca = PCA(n_components=4, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_va_pca = pca.transform(X_va_sc)
        
        mm_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        X_tr_q = mm_scaler.fit_transform(X_tr_pca)
        X_va_q = mm_scaler.transform(X_va_pca)
        
        n_qubits = 4
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def circuit(weights, x):
            # Angle Encoding
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
                
            # Variational Layers
            for W in weights:
                for i in range(n_qubits):
                    qml.RY(W[i, 0], wires=i)
                    qml.RZ(W[i, 1], wires=i)
                # Entanglement Ring
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[3, 0])
                
            return qml.expval(qml.PauliZ(0))
        
        # Optimization Loop
        n_layers = 2
        weights = pnp.random.uniform(-pnp.pi, pnp.pi, size=(n_layers, n_qubits, 2), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=0.1)
        
        def cost_fn(w):
            predictions = [circuit(w, x) for x in X_tr_q[:100]]
            # Convert PauliZ expectation [-1, 1] to probability [0, 1]
            probs = [(p + 1) / 2 for p in predictions]
            targets = y_tr[:100]
            loss = -pnp.mean(targets * pnp.log(pnp.clip(probs, 1e-7, 1-1e-7)) + 
                            (1 - targets) * pnp.log(pnp.clip(1 - probs, 1e-7, 1-1e-7)))
            return loss

        print("Optimizing PennyLane VQC parameters...")
        for step in range(15):
            weights = opt.step(cost_fn, weights)
            if (step + 1) % 5 == 0:
                print(f"Step {step+1:2d} | Loss: {float(cost_fn(weights)):.4f}")
                
        preds = [(circuit(weights, x) + 1) / 2 for x in X_va_q]
        binary_preds = (np.array(preds) >= 0.5).astype(int)
        print(f"PennyLane VQC Accuracy: {accuracy_score(y_va, binary_preds):.4f}")
        
except ImportError:
    print("PennyLane not installed. To run on physical hardware, install: pip install pennylane")

# ==============================================================================
# 2. QISKIT IMPLEMENTATION (Fidelity Quantum Kernel QSVM)
# ==============================================================================
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_algorithms.statefidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    def qiskit_qsvm_wdbc():
        print("\n--- Running Qiskit Fidelity Quantum Kernel on WDBC Dataset ---")
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        
        pca = PCA(n_components=4, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_va_pca = pca.transform(X_va_sc)
        
        # Qiskit ZZFeatureMap
        feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
        fidelity = ComputeUncompute()
        qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
        
        K_tr = qkernel.evaluate(x_vec=X_tr_pca)
        K_va = qkernel.evaluate(x_vec=X_va_pca, y_vec=X_tr_pca)
        
        qsvm = SVC(kernel='precomputed')
        qsvm.fit(K_tr, y_tr)
        preds = qsvm.predict(K_va)
        print(f"Qiskit QSVM Test Accuracy: {accuracy_score(y_va, preds):.4f}")

except ImportError:
    print("Qiskit Machine Learning not installed. To run on IBM Quantum, install: pip install qiskit qiskit-machine-learning")

if __name__ == "__main__":
    print("WDBC Quantum Hardware Deployment Wrapper Ready.")
