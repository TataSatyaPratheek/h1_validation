"""
COMPREHENSIVE MPS HIGH-QUBIT HIGH-BODY ANALYSIS
================================================
Test all problems with MPSCircuit at 500 qubits
with n-body correlations up to 20
"""
import numpy as np
import tensorcircuit as tc
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time
import warnings
warnings.filterwarnings('ignore')

tc.set_backend('numpy')  # Use numpy for MPS compatibility
tc.set_dtype('complex64')

print('='*70)
print('COMPREHENSIVE MPS HIGH-QUBIT HIGH-BODY ANALYSIS')
print('='*70)
print()


def create_mps_qfm(n_qubits, max_body=2, bond_dim=50):
    """
    Create QFM using MPSCircuit with n-body observables
    """
    def qfm_fn(inp):
        c = tc.MPSCircuit(n_qubits)
        c.set_split_rules({'max_singular_values': bond_dim})
        
        # Encoding (redundant for inputs < n_qubits)
        for i in range(n_qubits):
            c.ry(i, theta=float(inp[i % len(inp)]) * np.pi)
        
        # Chain entanglement
        for i in range(n_qubits - 1):
            c.cnot(i, i + 1)
        
        features = []
        
        # 1-body: Sample evenly across circuit
        sample_1body = list(range(0, n_qubits, max(1, n_qubits // 20)))[:20]
        for i in sample_1body:
            features.append(float(c.expectation_ps(z=[i]).real))
        
        # 2-body: Adjacent pairs (first 20)
        if max_body >= 2:
            for i in range(min(20, n_qubits - 1)):
                features.append(float(c.expectation_ps(z=[i, i+1]).real))
        
        # 3-body: Every 3rd qubit triplets
        if max_body >= 3:
            for i in range(min(15, n_qubits - 2)):
                features.append(float(c.expectation_ps(z=[i, i+1, i+2]).real))
        
        # 5-body: Sample 10 5-tuples
        if max_body >= 5:
            for i in range(min(10, n_qubits - 4)):
                indices = list(range(i, min(i+5, n_qubits)))
                if len(indices) == 5:
                    features.append(float(c.expectation_ps(z=indices).real))
        
        # 10-body: Sample 5 10-tuples
        if max_body >= 10:
            for i in range(min(5, n_qubits - 9)):
                indices = list(range(i, min(i+10, n_qubits)))
                if len(indices) == 10:
                    features.append(float(c.expectation_ps(z=indices).real))
        
        # 20-body: Sample 3 20-tuples
        if max_body >= 20:
            for i in range(min(3, n_qubits - 19)):
                indices = list(range(i, min(i+20, n_qubits)))
                if len(indices) == 20:
                    features.append(float(c.expectation_ps(z=indices).real))
        
        return np.array(features)
    
    return qfm_fn


def test_problem(name, X, y, n_qubits_list, max_body_list, bond_dim=50):
    """Test a problem across different qubit counts and body orders"""
    print(f'\n{"="*70}')
    print(f'PROBLEM: {name}')
    print(f'{"="*70}')
    
    n_samples = len(X)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    results = []
    
    for n_qubits in n_qubits_list:
        for max_body in max_body_list:
            if max_body > n_qubits:
                continue
            
            try:
                qfm = create_mps_qfm(n_qubits, max_body, bond_dim)
                
                start = time.time()
                F_train = np.array([qfm(x) for x in X_train])
                F_test = np.array([qfm(x) for x in X_test])
                elapsed = time.time() - start
                
                # QFM + Linear
                lr = LogisticRegression(max_iter=500)
                lr.fit(F_train, y_train)
                qfm_acc = lr.score(F_test, y_test) * 100
                
                # Classical baseline (RF on raw)
                rf = RandomForestClassifier(n_estimators=50)
                rf.fit(X_train, y_train)
                rf_acc = rf.score(X_test, y_test) * 100
                
                adv = qfm_acc - rf_acc
                marker = '✅' if adv > 0 else ''
                
                print(f'{n_qubits:>4}Q | {max_body:>2}-body | {F_train.shape[1]:>4} feat | '
                      f'QFM={qfm_acc:>5.1f}% | RF={rf_acc:>5.1f}% | Δ={adv:>+5.1f}% {marker}')
                
                results.append({
                    'n_qubits': n_qubits,
                    'max_body': max_body,
                    'n_features': F_train.shape[1],
                    'qfm_acc': qfm_acc,
                    'rf_acc': rf_acc,
                    'advantage': adv,
                    'time': elapsed
                })
                
            except Exception as e:
                print(f'{n_qubits:>4}Q | {max_body:>2}-body | FAILED: {str(e)[:40]}')
    
    return results


# =============================================================================
# PROBLEM 1: N-BIT PARITY
# =============================================================================
print('\n' + '='*70)
print('GENERATING PROBLEMS')
print('='*70)

# 8-bit parity
n_bits = 8
n_samples = 200
X_parity = np.random.randint(0, 2, (n_samples, n_bits)).astype(np.float32)
y_parity = (X_parity.sum(axis=1) % 2).astype(int)
print(f'Parity: {X_parity.shape}, Class balance: {y_parity.mean():.2f}')

# Network packets
def gen_network(n=200, bits=8):
    X, y = [], []
    for _ in range(n):
        data = np.random.randint(0, 2, bits - 1)
        parity = sum(data) % 2
        packet = np.append(data, parity).astype(np.float32)
        if np.random.random() < 0.5:
            packet[np.random.randint(bits)] = 1 - packet[np.random.randint(bits)]
            y.append(1)
        else:
            y.append(0)
        X.append(packet)
    return np.array(X), np.array(y)

X_network, y_network = gen_network(200, 8)
print(f'Network: {X_network.shape}, Class balance: {y_network.mean():.2f}')

# RAID parity
def gen_raid(n=200, drives=8):
    X, y = [], []
    for _ in range(n):
        data = np.random.randint(0, 2, drives - 1)
        parity = sum(data) % 2
        raid = np.append(data, parity).astype(np.float32)
        if np.random.random() < 0.5:
            raid[np.random.randint(drives)] = 1 - raid[np.random.randint(drives)]
            y.append(1)
        else:
            y.append(0)
        X.append(raid)
    return np.array(X), np.array(y)

X_raid, y_raid = gen_raid(200, 8)
print(f'RAID: {X_raid.shape}, Class balance: {y_raid.mean():.2f}')

# Majority vote
X_majority = np.random.randint(0, 2, (200, 8)).astype(np.float32)
y_majority = (X_majority.sum(axis=1) >= 4).astype(int)
print(f'Majority: {X_majority.shape}, Class balance: {y_majority.mean():.2f}')

# =============================================================================
# RUN TESTS
# =============================================================================
qubit_counts = [8, 50, 100, 200, 500]
body_orders = [2, 5, 10, 20]

all_results = {}

all_results['parity'] = test_problem('8-BIT PARITY', X_parity, y_parity, 
                                      qubit_counts, body_orders)

all_results['network'] = test_problem('NETWORK PACKET VALIDATION', X_network, y_network,
                                       qubit_counts, body_orders)

all_results['raid'] = test_problem('RAID PARITY', X_raid, y_raid,
                                    qubit_counts, body_orders)

all_results['majority'] = test_problem('MAJORITY VOTE', X_majority, y_majority,
                                        qubit_counts, body_orders)

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*70)
print('COMPREHENSIVE SUMMARY')
print('='*70)

print('\nBest QFM configurations by problem:')
for prob_name, results in all_results.items():
    if results:
        best = max(results, key=lambda x: x['advantage'])
        print(f"  {prob_name}: {best['n_qubits']}Q, {best['max_body']}-body, "
              f"Advantage={best['advantage']:+.1f}%")

print('\nEffect of n-body order (averaged across problems):')
for body in body_orders:
    advs = []
    for results in all_results.values():
        for r in results:
            if r['max_body'] == body:
                advs.append(r['advantage'])
    if advs:
        print(f"  {body:>2}-body: Avg advantage = {np.mean(advs):+.1f}%")

print('\nEffect of qubit count (averaged across problems):')
for n_q in qubit_counts:
    advs = []
    for results in all_results.values():
        for r in results:
            if r['n_qubits'] == n_q:
                advs.append(r['advantage'])
    if advs:
        print(f"  {n_q:>3}Q: Avg advantage = {np.mean(advs):+.1f}%")
