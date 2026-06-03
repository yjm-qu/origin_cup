"""
训练代码 v6B
基于设计文档v6.0 方案B：卷积+二次池化+全局池化+扩展预处理
结合老师建议修改：
1. 损失函数: MSE Loss (y - P)^2
2. 量子电路: 重复8次 [RY(encoding) -> CRZ(theta) -> RY(theta)]
3. 编码公式: angle = (π/16) × value
"""
import sys
sys.path.insert(0, '/home/ubuntu/UnifiedQuantum-main/UnifiedQuantum-main')

import numpy as np
import time
import torch
import torch.nn as nn
from uniqc.circuit_builder import Circuit
from uniqc.simulator import Simulator


# ============ 超参数 ============
NUM_QUBITS = 8
NUM_LAYERS = 8

# 电路结构: 每层 [RY(encoding) -> CRZ(theta) -> RY(theta)]
NUM_CRZ_PER_LAYER = NUM_QUBITS - 1  # 7个相邻CRZ门
NUM_RY_PER_LAYER = NUM_QUBITS       # 8个RY门
NUM_PARAMS_PER_LAYER = NUM_CRZ_PER_LAYER + NUM_RY_PER_LAYER  # 15
NUM_PARAMS = NUM_PARAMS_PER_LAYER * NUM_LAYERS  # 120

EPOCHS = 10
BATCH_SIZE = 50
LEARNING_RATE = 0.001

BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8


# ============ 方案B：卷积+池化+通道分组池化+扩展 ============
class ConvPreprocessorB(nn.Module):
    """
    方案B：卷积+池化+通道分组池化+扩展
    输出：8维特征

    数据流：
    16×16×1 → Conv(1→4) → 16×16×4 → Pool → 8×8×4
    → 按通道分组：4个通道分成2组（每组2个）
    → 组内平均池化：每组 → 2个值
    → 复制扩展：2组×2值 × 2 = 8个值
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),  # 16×16×1 → 16×16×4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 16×16×4 → 8×8×4
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 16, 16)
        x = x.unsqueeze(1)  # (batch, 1, 16, 16)
        x = self.conv(x)    # (batch, 4, 8, 8)

        # 按通道分组池化：4个通道分成2组（每组2个通道）
        results = []
        for i in range(2):  # 2组
            # 取第i组通道 (每组2个通道)
            channel_group = x[:, i*2:(i+1)*2, :, :]  # (batch, 2, 8, 8)
            # 对空间维度做平均池化
            pooled = channel_group.mean(dim=(2, 3))  # (batch, 2)
            results.append(pooled)

        x = torch.cat(results, dim=1)  # (batch, 4)

        # 复制扩展到8维：4维 → 8维
        x = torch.cat([x, x], dim=1)  # (batch, 8)

        return x


# ============ 数据加载与预处理 ============
def load_and_preprocess_data():
    """加载数据并进行卷积预处理"""
    train_data = np.load('/home/ubuntu/mnist_train_1000_16_16.npz')
    x_train = train_data['data']  # shape: (1000, 16, 16)
    y_train = train_data['label']  # shape: (1000,)

    # 使用PyTorch进行卷积预处理
    preprocessor = ConvPreprocessorB()
    preprocessor.eval()

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_train).float()
        x_processed = preprocessor(x_tensor).numpy()

    return x_processed, y_train


# ============ 量子线路 ============
def build_circuit(encoded_data, params):
    """
    构建量子线路：重复8次 [RY(encoding) -> CRZ(theta) -> RY(theta)] + 测量
    """
    circuit = Circuit(NUM_QUBITS)

    # ===== 重复8次 =====
    for layer in range(NUM_LAYERS):
        # ---- 1. RY(encoding) 角度编码层 ----
        for i in range(NUM_QUBITS):
            value = encoded_data[i]
            angle = (np.pi / 16) * value  # 角度编码: angle = (π/16) × value
            circuit.ry(i, angle)

        # ---- 2. CRZ(theta) 可训练纠缠层 ----
        base_idx = layer * NUM_PARAMS_PER_LAYER
        crz_idx = base_idx
        for i in range(NUM_CRZ_PER_LAYER):
            theta = params[crz_idx + i]
            circuit.crz(i, i + 1, theta)

        # ---- 3. RY(theta) 可训练层 ----
        ry_base_idx = base_idx + NUM_CRZ_PER_LAYER
        for i in range(NUM_RY_PER_LAYER):
            theta = params[ry_base_idx + i]
            circuit.ry(i, theta)

    # ===== 测量层 =====
    circuit.measure(0)

    return circuit


def run_circuit(encoded_data, params):
    """运行量子线路，返回P(|0⟩)"""
    circuit = build_circuit(encoded_data, params)
    simulator = Simulator(backend_type="statevector")

    probs = simulator.simulate_pmeasure(circuit)

    # 计算第0个量子比特为0的概率
    prob_0 = 0.0
    for idx, p in enumerate(probs):
        if (idx & 1) == 0:  # 第0位为0
            prob_0 += p

    return prob_0


# ============ 损失和梯度 ============
def compute_loss(F, target):
    """损失函数: MSE Loss O = (y - P)^2"""
    return (target - F) ** 2


def compute_gradient(encoded_data, target, params):
    """使用参数移位法则计算梯度 (MSE Loss版本)"""
    grad = np.zeros(NUM_PARAMS)
    shift = np.pi / 2

    # 获得当前输出
    F_current = run_circuit(encoded_data, params)

    # MSE梯度: dL/dP = -2(P - y)
    coefficient = -2 * (F_current - target)

    for i in range(NUM_PARAMS):
        # 正向偏移
        params_plus = params.copy()
        params_plus[i] += shift
        F_plus = run_circuit(encoded_data, params_plus)

        # 负向偏移
        params_minus = params.copy()
        params_minus[i] -= shift
        F_minus = run_circuit(encoded_data, params_minus)

        # 链式法则
        grad[i] = coefficient * (F_plus - F_minus) / 2

    return grad


# ============ 训练 ============
def train():
    print("=" * 60)
    print("训练配置 - 方案B (卷积+二次池化+全局池化+扩展)")
    print("=" * 60)
    print(f"量子比特数: {NUM_QUBITS}")
    print(f"Ansatz层数: {NUM_LAYERS}")
    print(f"参数数量: {NUM_PARAMS}")
    print(f"训练轮数: {EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print("=" * 60)

    print("\n加载并预处理数据...")
    x_train, y_train = load_and_preprocess_data()
    print(f"训练样本数: {len(x_train)}")
    print(f"预处理后数据维度: {x_train.shape[1]}")
    print(f"标签分布: 0(数字4): {np.sum(y_train==0)}, 1(数字9): {np.sum(y_train==1)}")
    print(f"数据范围: [{x_train.min():.4f}, {x_train.max():.4f}]")

    # 初始化参数
    np.random.seed(42)
    params = np.random.randn(NUM_PARAMS) * 0.1

    # ADAM优化器状态
    m = np.zeros(NUM_PARAMS)
    v = np.zeros(NUM_PARAMS)
    t = 0

    start_time = time.time()
    last_report_time = start_time

    print("\n开始训练...")
    print("=" * 60)

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # 打乱数据
        indices = np.random.permutation(len(x_train))

        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(x_train), BATCH_SIZE):
            batch_idx = indices[batch_start:batch_start + BATCH_SIZE]

            # 计算批次梯度
            batch_grad = np.zeros(NUM_PARAMS)
            batch_loss = 0.0

            for idx in batch_idx:
                encoded_data = x_train[idx]
                target = y_train[idx]

                # 计算梯度
                sample_grad = compute_gradient(encoded_data, target, params)
                batch_grad += sample_grad

                # 计算损失
                F = run_circuit(encoded_data, params)
                loss = compute_loss(F, target)
                batch_loss += loss

            # 平均梯度
            batch_grad /= len(batch_idx)
            batch_loss /= len(batch_idx)

            # ADAM更新
            t += 1
            m = BETA1 * m + (1 - BETA1) * batch_grad
            v = BETA2 * v + (1 - BETA2) * (batch_grad ** 2)
            m_hat = m / (1 - BETA1 ** t)
            v_hat = v / (1 - BETA2 ** t)
            params = params - LEARNING_RATE * m_hat / (np.sqrt(v_hat) + EPSILON)

            total_loss += batch_loss
            n_batches += 1

            # 每60秒汇报进度
            current_time = time.time()
            if current_time - last_report_time >= 60:
                elapsed = current_time - start_time
                progress = (epoch * len(x_train) + batch_start) / (EPOCHS * len(x_train)) * 100
                print(f"[{elapsed:.0f}秒] 进度: {progress:.1f}% (Epoch {epoch+1}/{EPOCHS}, Batch {batch_start//BATCH_SIZE + 1})")
                last_report_time = current_time

        # 本轮统计
        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1}/{EPOCHS}, 损失: {avg_loss:.4f}, 时间: {epoch_time:.1f}秒")

    total_time = time.time() - start_time

    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总训练时间: {total_time:.1f}秒")

    # 保存参数
    np.save('/home/ubuntu/origin_cup/params_v6B_mse.npy', params)
    print(f"参数已保存到 params_v6B_mse.npy")


if __name__ == "__main__":
    train()
