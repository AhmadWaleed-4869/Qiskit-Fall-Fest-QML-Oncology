import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

# Set random seed
SEED = 42
np.random.seed(SEED)

def generate_synthetic_metabric(n_samples=500, n_genomic=24):
    age = np.random.normal(60, 12, n_samples)
    tumor_size = np.random.exponential(20, n_samples)
    lymph_nodes = np.random.poisson(2, n_samples)
    nottingham_index = np.random.uniform(2, 6, n_samples)
    
    er_status = np.random.choice(['Positive', 'Negative'], p=[0.75, 0.25], size=n_samples)
    pr_status = np.random.choice(['Positive', 'Negative'], p=[0.65, 0.35], size=n_samples)
    her2_status = np.random.choice(['Positive', 'Negative'], p=[0.80, 0.20], size=n_samples)
    pam50_subtype = np.random.choice(['LumA', 'LumB', 'Her2', 'Basal', 'Normal'], p=[0.4, 0.25, 0.15, 0.15, 0.05], size=n_samples)
    chemo_treatment = np.random.choice(['YES', 'NO'], p=[0.20, 0.80], size=n_samples)
    hormone_treatment = np.random.choice(['YES', 'NO'], p=[0.60, 0.40], size=n_samples)
    
    genomic_features = np.random.normal(0, 1, size=(n_samples, n_genomic))
    genomic_cols = [f"Gene_Expr_{i+1}" for i in range(n_genomic)]
    
    logit = (0.03 * (60 - age) - 0.02 * tumor_size - 0.15 * lymph_nodes + 
             (1.0 if er_status[0] == 'Positive' else -0.5) + genomic_features[:, 0] * 0.5 - genomic_features[:, 1] * 0.4)
    prob = 1 / (1 + np.exp(-logit))
    target = (prob > np.median(prob)).astype(int)
    
    df_dict = {
        'Age': age,
        'Tumor_Size': tumor_size,
        'Lymph_Nodes': lymph_nodes,
        'Nottingham_Index': nottingham_index,
        'ER_Status': er_status,
        'PR_Status': pr_status,
        'HER2_Status': her2_status,
        'PAM50_Subtype': pam50_subtype,
        'Chemotherapy': chemo_treatment,
        'Hormone_Therapy': hormone_treatment,
    }
    for i, col in enumerate(genomic_cols):
        df_dict[col] = genomic_features[:, i]
        
    df = pd.DataFrame(df_dict)
    mask_ts = np.random.rand(n_samples) < 0.05
    df.loc[mask_ts, 'Tumor_Size'] = np.nan
    return df, target

X_raw, y = generate_synthetic_metabric(n_samples=500)

def preprocess_fold(X_train_df, X_val_df):
    cat_cols = X_train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    num_imputer = SimpleImputer(strategy='median')
    X_train_num = num_imputer.fit_transform(X_train_df[num_cols])
    X_val_num = num_imputer.transform(X_val_df[num_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train_cat_imp = cat_imputer.fit_transform(X_train_df[cat_cols])
    X_val_cat_imp = cat_imputer.transform(X_val_df[cat_cols])
    
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    X_train_cat = ohe.fit_transform(X_train_cat_imp)
    X_val_cat = ohe.transform(X_val_cat_imp)
    
    X_train_full = np.hstack([X_train_num, X_train_cat])
    X_val_full = np.hstack([X_val_num, X_val_cat])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_val_scaled = scaler.transform(X_val_full)
    
    pca_dict = {}
    for n_comp in [12, 4, 2]:
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_tr_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        pca_dict[n_comp] = (X_tr_pca, X_val_pca)
        
    return X_train_scaled, X_val_scaled, pca_dict

class QuantumStateSimulator:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.reset()
        
    def reset(self):
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0

    def apply_ry(self, qubit_idx, angle):
        c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
        ry_matrix = np.array([[c, -s], [s, c]], dtype=complex)
        self._apply_single_qubit_gate(qubit_idx, ry_matrix)

    def apply_rz(self, qubit_idx, angle):
        rz_matrix = np.array([[np.exp(-1j * angle / 2.0), 0], [0, np.exp(1j * angle / 2.0)]], dtype=complex)
        self._apply_single_qubit_gate(qubit_idx, rz_matrix)

    def _apply_single_qubit_gate(self, qubit_idx, gate_mat):
        shape = [2] * self.n_qubits
        tensor = self.state.reshape(shape)
        tensor = np.swapaxes(tensor, 0, qubit_idx)
        new_tensor = np.tensordot(gate_mat, tensor, axes=(1, 0))
        new_tensor = np.swapaxes(new_tensor, 0, qubit_idx)
        self.state = new_tensor.flatten()

    def apply_cnot(self, control, target):
        shape = [2] * self.n_qubits
        tensor = self.state.reshape(shape)
        slices_control_1 = [slice(None)] * self.n_qubits
        slices_control_1[control] = 1
        sub_tensor = tensor[tuple(slices_control_1)]
        sub_tensor_flipped = np.flip(sub_tensor, axis=target if target < control else target - 1)
        tensor[tuple(slices_control_1)] = sub_tensor_flipped
        self.state = tensor.flatten()

    def measure_pauli_z(self, qubit_idx):
        probs = np.abs(self.state)**2
        shape = [2] * self.n_qubits
        probs_tensor = probs.reshape(shape)
        slices_0 = [slice(None)] * self.n_qubits
        slices_0[qubit_idx] = 0
        p0 = np.sum(probs_tensor[tuple(slices_0)])
        slices_1 = [slice(None)] * self.n_qubits
        slices_1[qubit_idx] = 1
        p1 = np.sum(probs_tensor[tuple(slices_1)])
        return p0 - p1

def compute_quantum_kernel_matrix(X1, X2, n_qubits=4):
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    sim1 = QuantumStateSimulator(n_qubits)
    sim2 = QuantumStateSimulator(n_qubits)
    
    states_X1 = []
    for i in range(n1):
        sim1.reset()
        for q in range(n_qubits):
            sim1.apply_ry(q, X1[i, q])
            sim1.apply_rz(q, X1[i, q])
        states_X1.append(sim1.state.copy())
        
    states_X2 = []
    for j in range(n2):
        sim2.reset()
        for q in range(n_qubits):
            sim2.apply_ry(q, X2[j, q])
            sim2.apply_rz(q, X2[j, q])
        states_X2.append(sim2.state.copy())
        
    for i in range(n1):
        for j in range(n2):
            overlap = np.vdot(states_X1[i], states_X2[j])
            K[i, j] = np.abs(overlap)**2
            
    return K

class NumPyVQC:
    def __init__(self, n_qubits=4, n_layers=2, lr=0.05, n_epochs=15):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = np.random.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 2))
        self.bias = 0.0
        
    def _forward_sample(self, x, weights):
        sim = QuantumStateSimulator(self.n_qubits)
        sim.reset()
        for q in range(self.n_qubits):
            sim.apply_ry(q, x[q])
            
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                sim.apply_ry(q, weights[l, q, 0])
                sim.apply_rz(q, weights[l, q, 1])
            for q in range(self.n_qubits):
                sim.apply_cnot(q, (q + 1) % self.n_qubits)
                
        exp_z = sim.measure_pauli_z(0)
        prob = 1 / (1 + np.exp(-(exp_z + self.bias)))
        return np.clip(prob, 1e-7, 1 - 1e-7)

    def fit(self, X, y):
        n_samples = X.shape[0]
        for epoch in range(self.n_epochs):
            grad_w = np.zeros_like(self.weights)
            eps = 1e-2
            sub_idx = np.random.choice(n_samples, min(40, n_samples), replace=False)
            
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    for k in range(2):
                        w_plus = self.weights.copy()
                        w_plus[l, q, k] += eps
                        p_plus = np.array([self._forward_sample(X[i], w_plus) for i in sub_idx])
                        
                        w_minus = self.weights.copy()
                        w_minus[l, q, k] -= eps
                        p_minus = np.array([self._forward_sample(X[i], w_minus) for i in sub_idx])
                        
                        l_plus = -np.mean(y[sub_idx] * np.log(p_plus) + (1 - y[sub_idx]) * np.log(1 - p_plus))
                        l_minus = -np.mean(y[sub_idx] * np.log(p_minus) + (1 - y[sub_idx]) * np.log(1 - p_minus))
                        grad_w[l, q, k] = (l_plus - l_minus) / (2 * eps)
                        
            self.weights -= self.lr * grad_w

    def predict_proba(self, X):
        probs = np.array([self._forward_sample(X[i], self.weights) for i in range(X.shape[0])])
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# 5-Fold Cross-Validation Evaluation Pipeline
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = []

print("Running 5-Fold Cross Validation Benchmark...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
    print(f"\n--- Fold {fold+1} ---")
    X_tr_df, X_va_df = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]
    
    X_tr_full, X_va_full, pca_dict = preprocess_fold(X_tr_df, X_va_df)
    
    # 1. Classical Full Feature Models
    models_full = {
        'Logistic Regression (Full 63)': LogisticRegression(max_iter=1000, random_state=SEED),
        'SVM RBF (Full 63)': SVC(probability=True, random_state=SEED),
        'Random Forest (Full 63)': RandomForestClassifier(n_estimators=100, random_state=SEED),
        'XGBoost (Full 63)': xgb.XGBClassifier(eval_metric='logloss', random_state=SEED),
        'LightGBM (Full 63)': lgb.LGBMClassifier(random_state=SEED, verbose=-1),
        'MLP Neural Net (Full 63)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED)
    }
    
    for name, model in models_full.items():
        t0 = time.time()
        model.fit(X_tr_full, y_tr)
        t_fit = time.time() - t0
        preds = model.predict(X_va_full)
        probs = model.predict_proba(X_va_full)[:, 1]
        
        results.append({
            'Fold': fold+1, 'Representation': 'Full (63)', 'Model': name,
            'Accuracy': accuracy_score(y_va, preds),
            'F1_Score': f1_score(y_va, preds),
            'ROC_AUC': roc_auc_score(y_va, probs),
            'Log_Loss': log_loss(y_va, probs),
            'Runtime_Sec': t_fit
        })
        
    # 2. PCA-12 Classical Models & Quantum Models
    X_tr_p12, X_va_p12 = pca_dict[12]
    models_pca12 = {
        'Logistic Regression (PCA-12)': LogisticRegression(max_iter=1000, random_state=SEED),
        'SVM RBF (PCA-12)': SVC(probability=True, random_state=SEED),
        'XGBoost (PCA-12)': xgb.XGBClassifier(eval_metric='logloss', random_state=SEED),
        'MLP Neural Net (PCA-12)': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=SEED)
    }
    
    for name, model in models_pca12.items():
        t0 = time.time()
        model.fit(X_tr_p12, y_tr)
        t_fit = time.time() - t0
        preds = model.predict(X_va_p12)
        probs = model.predict_proba(X_va_p12)[:, 1]
        
        results.append({
            'Fold': fold+1, 'Representation': 'PCA-12', 'Model': name,
            'Accuracy': accuracy_score(y_va, preds),
            'F1_Score': f1_score(y_va, preds),
            'ROC_AUC': roc_auc_score(y_va, probs),
            'Log_Loss': log_loss(y_va, probs),
            'Runtime_Sec': t_fit
        })
        
    # 3. PCA-4 Quantum & Classical Models
    X_tr_p4, X_va_p4 = pca_dict[4]
    
    # Quantum VQC (4 Qubits)
    vqc4 = NumPyVQC(n_qubits=4, n_layers=2, n_epochs=10)
    t0 = time.time()
    vqc4.fit(X_tr_p4, y_tr)
    t_fit = time.time() - t0
    preds_vqc4 = vqc4.predict(X_va_p4)
    probs_vqc4 = vqc4.predict_proba(X_va_p4)[:, 1]
    
    results.append({
        'Fold': fold+1, 'Representation': 'PCA-4 (Quantum)', 'Model': 'VQC / QNN (4 Qubits)',
        'Accuracy': accuracy_score(y_va, preds_vqc4),
        'F1_Score': f1_score(y_va, preds_vqc4),
        'ROC_AUC': roc_auc_score(y_va, probs_vqc4),
        'Log_Loss': log_loss(y_va, probs_vqc4),
        'Runtime_Sec': t_fit
    })

    # Quantum Support Vector Machine (QSVM Kernel 4 Qubits)
    t0 = time.time()
    # Downsample for kernel speed in simulation
    idx_sub_tr = np.random.choice(X_tr_p4.shape[0], min(80, X_tr_p4.shape[0]), replace=False)
    K_tr = compute_quantum_kernel_matrix(X_tr_p4[idx_sub_tr], X_tr_p4[idx_sub_tr], n_qubits=4)
    K_va = compute_quantum_kernel_matrix(X_va_p4, X_tr_p4[idx_sub_tr], n_qubits=4)
    
    qsvm = SVC(kernel='precomputed', probability=True, random_state=SEED)
    qsvm.fit(K_tr, y_tr[idx_sub_tr])
    t_fit = time.time() - t0
    preds_qsvm = qsvm.predict(K_va)
    probs_qsvm = qsvm.predict_proba(K_va)[:, 1]
    
    results.append({
        'Fold': fold+1, 'Representation': 'PCA-4 (Quantum)', 'Model': 'QSVM Kernel (4 Qubits)',
        'Accuracy': accuracy_score(y_va, preds_qsvm),
        'F1_Score': f1_score(y_va, preds_qsvm),
        'ROC_AUC': roc_auc_score(y_va, probs_qsvm),
        'Log_Loss': log_loss(y_va, probs_qsvm),
        'Runtime_Sec': t_fit
    })

res_df = pd.DataFrame(results)

summary_df = res_df.groupby(['Representation', 'Model']).agg({
    'Accuracy': ['mean', 'std'],
    'F1_Score': ['mean', 'std'],
    'ROC_AUC': ['mean', 'std'],
    'Log_Loss': ['mean', 'std'],
    'Runtime_Sec': 'mean'
}).reset_index()

summary_df.columns = ['Representation', 'Model', 'Acc_Mean', 'Acc_Std', 'F1_Mean', 'F1_Std', 'AUC_Mean', 'AUC_Std', 'Loss_Mean', 'Loss_Std', 'Runtime_Sec']

print("\n" + "="*80)
print("FINAL EVALUATION BENCHMARK SUMMARY TABLE")
print("="*80)
print(summary_df.to_string(index=False))

# Save output table to scratch and out
summary_df.to_csv("/workspace/scratch/benchmark_summary.csv", index=False)
