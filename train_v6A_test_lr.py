"""
测试1：调整学习率
基于 train_v6A_test.py 修改
"""
import sys
import os
sys.path.insert(0, '/home/ubuntu/UnifiedQuantum-main/UnifiedQuantum-main')

os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import time
import torch
import torch.nn as nn
from uniqc.circuit_builder import Circuit
from uniqc.simulator import Simulator


# ============ 超参数 ============
NUM_QUBITS = 8
NUM_LAYERS = 8

NUM_CRZ_PER_LAYER = NUM_QUBITS - 1
NUM_RY_PER_LAYER = NUM_QUBITS
NUM_PARAMS_PER_LAYER = NUM_CRZ_PER_LAYER + NUM_RY_PER_LAYER
NUM_PARAMS = NUM_PARAMS_PER_LAYER * NUM_LAYERS

# 调整学习率
EPOCHS = 10
NUM_SAMPLES = 100
BATCH_SIZE = 10
# 关键修改：学习率降低10倍
LEARNING_RATE = 0.0001

BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8


# ============ 方案A：卷积+池化+分组池化预处理 ============
class ConvPreprocessorA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 16, 16)
        x = x.unsqueeze(1)
        x = self.conv(x)

        results = []
        for i in range(8):
            col = x[:, :, :, i]
            pooled = col.mean(dim=2)
            pooled = pooled.sum(dim=1)
            results.append(pooled)

        x = torch.stack(results, dim=1)
        return x


# ============ 数据加载与预处理 ============
def load_and_preprocess_data():
    train_data = np.load('/home/ubuntu/mnist_train_1000_16_16.npz')
    x_train = train_data['data']
    y_train = train_data['label']

    preprocessor = ConvPreprocessorA()
    preprocessor.eval()

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_train).float()
        x_processed = preprocessor(x_tensor).numpy()

    return x_processed, y_train


# ============ 量子线路 ============
def build_circuit(encoded_data, params):
    circuit = Circuit(NUM_QUBITS)

    for layer in range(NUM_LAYERS):
        for i in range(NUM_QUBITS):
            value = encoded_data[i]
            angle = (np.pi / 16) * value
            circuit.ry(i, angle)

        base_idx = layer * NUM_PARAMS_PER_LAYER
        crz_idx = base_idx
        for i in range(NUM_CRZ_PER_LAYER):
            theta = params[crz_idx + i]
            circuit.crz(i, i + 1, theta)

        ry_base_idx = base_idx + NUM_CRZ_PER_LAYER
        for i in range(NUM_RY_PER_LAYER):
            theta = params[ry_base_idx + i]
            circuit.ry(i, theta)

    circuit.measure(0)

    return circuit


def run_circuit(encoded_data, params):
    circuit = build_circuit(encoded_data, params)
    simulator = Simulator(backend_type="statevector")

    probs = simulator.simulate_pmeasure(circuit)

    prob_0 = 0.0
    for idx, p in enumerate(probs):
        if (idx & 1) == 0:
            prob_0 += p

    return prob_0


# ============ 损失和梯度 ============
def compute_loss(F, target):
    return (target - F) ** 2


def compute_gradient(encoded_data, target, params):
    grad = np.zeros(NUM_PARAMS)
    shift = np.pi / 2

    F_current = run_circuit(encoded_data, params)
    # 修复：coefficient 改成正确公式
    coefficient = 2 * (F_current - target)

    for i in range(NUM_PARAMS):
        params_plus = params.copy()
        params_plus[i] += shift
        F_plus = run_circuit(encoded_data, params_plus)

        params_minus = params.copy()
        params_minus[i] -= shift
        F_minus = run_circuit(encoded_data, params_minus)

        grad[i] = coefficient * (F_plus - F_minus) / 2

    return grad


# ============ 训练 ============
def train():
    print("=" * 60)
    print("测试1：调整学习率 (LR=0.0001)")
    print("=" * 60)
    print(f"量子比特数: {NUM_QUBITS}")
    print(f"Ansatz层数: {NUM_LAYERS}")
    print(f"参数数量: {NUM_PARAMS}")
    print(f"训练轮数: {EPOCHS}")
    print(f"样本数: {NUM_SAMPLES}")
    print(f"学习率: {LEARNING_RATE}")
    print("=" * 60)

    print("\n加载并预处理数据...")
    x_train, y_train = load_and_preprocess_data()

    x_train_subset = x_train[:NUM_SAMPLES]
    y_train_subset = y_train[:NUM_SAMPLES]

    print(f"训练样本数: {len(x_train_subset)}")
    print(f"标签分布: 0(数字4): {np.sum(y_train_subset==0)}, 1(数字9): {np.sum(y_train_subset==1)}")
    print(f"数据范围: [{x_train_subset.min():.4f}, {x_train_subset.max():.4f}]")

    np.random.seed(42)
    params = np.random.uniform(-np.pi, np.pi, NUM_PARAMS)
    print(f"\n参数初始化: 均匀分布 -π 到 π")
    print(f"参数范围: [{params.min():.4f}, {params.max():.4f}]")

    m = np.zeros(NUM_PARAMS)
    v = np.zeros(NUM_PARAMS)
    t = 0

    start_time = time.time()

    print("\n开始训练...")
    print("=" * 60)

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        indices = np.arange(len(x_train_subset))

        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(x_train_subset), BATCH_SIZE):
            batch_idx = indices[batch_start:batch_start + BATCH_SIZE]

            batch_grad = np.zeros(NUM_PARAMS)
            batch_loss = 0.0

            for idx in batch_idx:
                encoded_data = x_train_subset[idx]
                target = y_train_subset[idx]

                sample_grad = compute_gradient(encoded_data, target, params)
                batch_grad += sample_grad

                F = run_circuit(encoded_data, params)
                loss = compute_loss(F, target)
                batch_loss += loss

                elapsed = time.time() - start_time
                print(f"[{elapsed:.0f}秒] 样本 {idx+1}/{NUM_SAMPLES}, 损失: {loss:.4f}", flush=True)

            batch_grad /= len(batch_idx)
            batch_loss /= len(batch_idx)

            t += 1
            m = BETA1 * m + (1 - BETA1) * batch_grad
            v = BETA2 * v + (1 - BETA2) * (batch_grad ** 2)
            m_hat = m / (1 - BETA1 ** t)
            v_hat = v / (1 - BETA2 ** t)
            params = params - LEARNING_RATE * m_hat / (np.sqrt(v_hat) + EPSILON)

            total_loss += batch_loss
            n_batches += 1

        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1}/{EPOCHS}, 损失: {avg_loss:.4f}, 时间: {epoch_time:.1f}秒", flush=True)

    total_time = time.time() - start_time

    print("=" * 60)
    print("训练完成!")
    print(f"总训练时间: {total_time:.1f}秒")
    np.save('/home/ubuntu/origin_cup/params_test_lr.npy', params)


if __name__ == "__main__":
    import sys
    log_file = open('/home/ubuntu/origin_cup/train_test_lr_log.txt', 'w', buffering=1)

    class TeeOutput:
        def __init__(self, file):
            self.file = file
        def write(self, data):
            self.file.write(data)
            sys.__stdout__.write(data)
        def flush(self):
            self.file.flush()
            sys.__stdout__.flush()

    sys.stdout = TeeOutput(log_file)
    train()
