"""
训练代码 v5.2
基于老师建议修改：
1. 损失函数: MSE Loss (y - P)^2
2. 量子电路: 重复8次 [RY(encoding) -> CRZ(theta) -> RY(theta)]
"""
import sys
sys.path.insert(0, '/home/ubuntu/UnifiedQuantum-main/UnifiedQuantum-main')

import numpy as np
import time
from uniqc.circuit_builder import Circuit
from uniqc.simulator import Simulator


# ============ 超参数 ============
NUM_QUBITS = 8
NUM_LAYERS = 8  # 重复8次

# 电路结构: 每层 [RY(encoding) -> CRZ(theta) -> RY(theta)]
# CRZ: 7个相邻对 (0-1,1-2,...,6-7) 每对1个参数 = 7
# RY: 8个量子比特 × 1个参数 = 8
# 每层参数 = 7 + 8 = 15
# 8层总共 = 15 × 8 = 120 参数
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


# ============ 数据加载与预处理 ============
def load_and_preprocess_data():
    """加载数据并进行预处理"""
    train_data = np.load('/home/ubuntu/mnist_train_1000_16_16.npz')
    x_train = train_data['data']  # shape: (1000, 16, 16)
    y_train = train_data['label']  # shape: (1000,)

    # 标签已经是0和1：0→数字4, 1→数字9
    # 不需要转换

    # 行合并压缩（每2行求和，得到8个值）
    # 不做二值化，直接用原始浮点值求和
    x_compressed = np.zeros((1000, 8), dtype=np.float32)
    for i in range(8):
        x_compressed[:, i] = np.sum(x_train[:, i*2:(i+1)*2, :], axis=(1, 2))

    # x_compressed 范围是 0~16（每行16个像素，每个像素最大1.0）

    return x_compressed, y_train


# ============ 量子线路 ============
def build_circuit(encoded_data, params):
    """
    构建量子线路：重复8次 [RY(encoding) -> CRZ(theta) -> RY(theta)] + 测量

    电路结构（重复8次）:
    1. RY(encoding): 使用编码数据的RY门（角度编码）
    2. CRZ(theta): 可训练的CRZ纠缠门（相邻量子比特对）
    3. RY(theta): 可训练的RY门

    encoded_data: 8个整数值 (范围0~16)
    params: 120个可训练参数
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
        # 相邻量子比特对: (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)
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
    # P(|0⟩) = sum(P(|x⟩) for all x where bit 0 = 0)
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
    # 链式法则: dL/dtheta = dL/dP * dP/dtheta
    # = -2(P - y) * (dP/dtheta)
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

        # 链式法则: coefficient * dP/dtheta
        grad[i] = coefficient * (F_plus - F_minus) / 2

    return grad


# ============ 训练 ============
def train():
    print("=" * 60)
    print("训练配置")
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
    print(f"数据范围: [{x_train.min()}, {x_train.max()}]")

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
    np.save('/home/ubuntu/origin_cup/params_v5_mse.npy', params)
    print(f"参数已保存到 params_v5_mse.npy")


if __name__ == "__main__":
    train()
