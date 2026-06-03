"""
参数化量子线路模块
真正的参数化量子线路 + 数值梯度
"""
import sys
sys.path.insert(0, '/home/ubuntu/UnifiedQuantum-main/UnifiedQuantum-main')

import numpy as np
from uniqc.circuit_builder import Circuit
from uniqc.simulator import Simulator


NUM_QUBITS = 8
NUM_LAYERS = 5
NUM_PARAMS = 40


def build_circuit(data_vector, params):
    circuit = Circuit(NUM_QUBITS)
    for i in range(NUM_QUBITS):
        start_idx = i * 32
        chunk = data_vector[start_idx:start_idx + 32]
        chunk_norm = np.linalg.norm(chunk)
        if chunk_norm > 1e-10:
            angle = np.arccos(np.clip(chunk[0] / chunk_norm, -1, 1))
        else:
            angle = 0
        circuit.ry(i, angle)

    for layer in range(NUM_LAYERS):
        for qubit in range(NUM_QUBITS):
            idx = layer * NUM_QUBITS + qubit
            circuit.ry(qubit, params[idx])
        if layer % 2 == 0:
            for i in range(0, NUM_QUBITS - 1, 2):
                circuit.cnot(i, i + 1)
        else:
            for i in range(1, NUM_QUBITS - 1, 2):
                circuit.cnot(i, i + 1)

    circuit.measure(0)
    return circuit


def run_circuit(circuit):
    simulator = Simulator(backend_type="statevector")
    probs = simulator.simulate_pmeasure(circuit)
    prob_dict = {}
    for idx in range(len(probs)):
        bit_str = format(idx, f'0{NUM_QUBITS}b')
        prob_dict[bit_str] = probs[idx]
    prob_0 = sum(val for bit_str, val in prob_dict.items() if bit_str[NUM_QUBITS - 1] == '0')
    return prob_0


def compute_loss(prob, target):
    eps = 1e-8
    prob = np.clip(prob, eps, 1 - eps)
    return -target * np.log(prob) - (1 - target) * np.log(1 - prob)


def train():
    from utils.data import prepare_data

    print("Loading data...")
    x_train, y_train, x_test, y_test = prepare_data()
    print(f"Train: {len(x_train)}, Test: {len(x_test)}")

    params = np.random.randn(NUM_PARAMS) * 0.1

    EPOCHS = 30
    LR = 0.1
    BATCH_SIZE = 50

    print(f"Training for {EPOCHS} epochs, batch_size={BATCH_SIZE}...")

    for epoch in range(EPOCHS):
        indices = np.random.permutation(len(x_train))
        total_loss = 0

        for i in range(0, len(x_train), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]

            grad = np.zeros(NUM_PARAMS)
            batch_loss = 0

            for idx in batch_idx:
                data = x_train[idx]
                target = y_train[idx]

                eps = 0.2
                for j in range(NUM_PARAMS):
                    params_plus = params.copy()
                    params_plus[j] += eps
                    loss_plus = compute_loss(run_circuit(build_circuit(data, params_plus)), target)

                    params_minus = params.copy()
                    params_minus[j] -= eps
                    loss_minus = compute_loss(run_circuit(build_circuit(data, params_minus)), target)

                    grad[j] += (loss_plus - loss_minus) / (2 * eps)

                prob = run_circuit(build_circuit(data, params))
                batch_loss += compute_loss(prob, target)

            grad /= len(batch_idx)
            batch_loss /= len(batch_idx)

            params -= LR * grad
            total_loss += batch_loss

        avg_loss = total_loss / (len(x_train) / BATCH_SIZE)

        correct = 0
        for i in range(len(x_train)):
            prob = run_circuit(build_circuit(x_train[i], params))
            pred = 0 if prob > 0.5 else 1
            if pred == y_train[i]:
                correct += 1
        acc = correct / len(x_train)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Train Acc: {acc:.4f}")

    print("Training complete!")
    np.save("/home/ubuntu/origin_cup/params_v2.npy", params)

    correct = 0
    for i in range(len(x_test)):
        prob = run_circuit(build_circuit(x_test[i], params))
        pred = 0 if prob > 0.5 else 1
        if pred == y_test[i]:
            correct += 1

    accuracy = correct / len(x_test)
    Q = NUM_QUBITS
    P = NUM_PARAMS
    score = 44 * accuracy + 3 * ((8 - Q) / 7 + (100 - P) / 99)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("=== Results ===")
    print("Score,Acc,Q,P")
    print(f"{score:.2f},{accuracy:.2f},{Q},{P}")


if __name__ == "__main__":
    train()
