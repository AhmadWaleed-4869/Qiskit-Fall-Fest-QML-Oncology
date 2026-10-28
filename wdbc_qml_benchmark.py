import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

SEED = 42
np.random.seed(SEED)

# ==============================================================================
# 1. LOAD WISCONSIN DIAGNOSTIC BREAST CANCER (WDBC) DATASET
# ==============================================================================
def load_wdbc_data():
    """Loads 569 samples with 30 continuous cell nucleus morphological features."""
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names

# ==============================================================================
# 2. QUANTUM STATE SIMULATOR (NumPy Engine for Quantum Kernels)
# ==============================================================================
class QuantumStateSimulator:
    """State vector simulator implementing parameterized angle rotation gates."""
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.reset()
        
    def reset(self):
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0

    def apply_ry(self, qubit_idx, angle):
        c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
        self._apply_single_qubit_gate(qubit_idx, np.array([[c, -s], [s, c]], dtype=complex))

    def apply_rz(self, qubit_idx, angle):
        self._apply_single_qubit_gate(qubit_idx, np.array([[np.exp(-1j*angle/2), 0], [0, np.exp(1j*angle/2)]], dtype=complex))

    def _apply_single_qubit_gate(self, qubit_idx, gate_mat):
        shape = [2] * self.n_qubits
        tensor = self.state.reshape(shape)
        tensor = np.swapaxes(tensor, 0, qubit_idx)
        new_tensor = np.tensordot(gate_mat, tensor, axes=(1, 0))
        self.state = np.swapaxes(new_tensor, 0, qubit_idx).flatten()

def compute_quantum_fidelity_kernel(X1, X2, n_qubits=4):
    """Computes Quantum Fidelity Kernel Matrix: K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2"""
    n1, n2 = X1.shape[0], X2.shape[0]
    states1 = []
    sim = QuantumStateSimulator(n_qubits)
    
    for i in range(n1):
        sim.reset()
        for q in range(n_qubits):
            sim.apply_ry(q, X1[i, q])
            sim.apply_rz(q, X1[i, q] * 2.0)
        states1.append(sim.state.copy())
        
    states2 = []
    for j in range(n2):
        sim.reset()
        for q in range(n_qubits):
            sim.apply_ry(q, X2[j, q])
            sim.apply_rz(q, X2[j, q] * 2.0)
        states2.append(sim.state.copy())
        
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = np.abs(np.vdot(states1[i], states2[j]))**2
    return K

# ==============================================================================
# 3. BENCHMARK EXECUTION (5-Fold Stratified Cross-Validation)
# ==============================================================================
def run_wdbc_benchmark():
    X, y, _ = load_wdbc_data()
    print(f"Loaded WDBC dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        # Standard scaling
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        
        # PCA to 4 components (~79.3% variance retained)
        pca4 = PCA(n_components=4, random_state=SEED)
        X_tr_pca4 = pca4.fit_transform(X_tr_sc)
        X_va_pca4 = pca4.transform(X_va_sc)
        
        # MinMax Scaling to [-pi, pi] for angle encoding
        mm_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        X_tr_q4 = mm_scaler.fit_transform(X_tr_pca4)
        X_va_q4 = mm_scaler.transform(X_va_pca4)
        
        # Classical Benchmark Suite
        models = {
            'Full Classical - SVM (RBF)': SVC(probability=True, random_state=SEED).fit(X_tr_sc, y_tr),
            'Full Classical - Logistic Reg': LogisticRegression(random_state=SEED).fit(X_tr_sc, y_tr),
            'Full Classical - Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_tr_sc, y_tr),
            'PCA-4 Classical - SVM (RBF)': SVC(probability=True, random_state=SEED).fit(X_tr_pca4, y_tr),
            'PCA-4 Classical - Logistic Reg': LogisticRegression(random_state=SEED).fit(X_tr_pca4, y_tr),
            'PCA-4 Classical - KNN (k=5)': KNeighborsClassifier(n_neighbors=5).fit(X_tr_pca4, y_tr)
        }
        
        for name, model in models.items():
            t0 = time.time()
            data_va = X_va_sc if 'Full' in name else X_va_pca4
            preds = model.predict(data_va)
            probs = model.predict_proba(data_va)[:, 1]
            elapsed = time.time() - t0
            
            results.append({
                'Fold': fold, 'Model': name,
                'Accuracy': accuracy_score(y_va, preds),
                'F1': f1_score(y_va, preds),
                'AUC': roc_auc_score(y_va, probs),
                'Time': elapsed
            })
            
        # Quantum Fidelity Kernel SVM (QSVM)
        t0 = time.time()
        K_tr = compute_quantum_fidelity_kernel(X_tr_q4, X_tr_q4, n_qubits=4)
        K_va = compute_quantum_fidelity_kernel(X_va_q4, X_tr_q4, n_qubits=4)
        
        qsvm = SVC(kernel='precomputed', probability=True, random_state=SEED)
        qsvm.fit(K_tr, y_tr)
        q_preds = qsvm.predict(K_va)
        q_probs = qsvm.predict_proba(K_va)[:, 1]
        elapsed = time.time() - t0
        
        results.append({
            'Fold': fold, 'Model': 'PCA-4 Quantum - QSVM Kernel',
            'Accuracy': accuracy_score(y_va, q_preds),
            'F1': f1_score(y_va, q_preds),
            'AUC': roc_auc_score(y_va, q_probs),
            'Time': elapsed
        })
            
    df_res = pd.DataFrame(results)
    
    print("\n================================================================================")
    print("WISCONSIN BREAST CANCER (WDBC) BENCHMARK SUMMARY")
    print("================================================================================")
    for model_name, group in df_res.groupby('Model'):
        acc_m, acc_s = group['Accuracy'].mean(), group['Accuracy'].std()
        f1_m, f1_s = group['F1'].mean(), group['F1'].std()
        auc_m, auc_s = group['AUC'].mean(), group['AUC'].std()
        print(f"{model_name:<32} | Acc: {acc_m:.4f} ± {acc_s:.4f} | F1: {f1_m:.4f} ± {f1_s:.4f} | AUC: {auc_m:.4f} ± {auc_s:.4f}")
        
    return df_res

if __name__ == "__main__":
    run_wdbc_benchmark()
