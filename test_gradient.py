"""
梯度验证测试
使用数值微分来验证梯度计算是否正确
"""
import sys
sys.path.insert(0, '/home/ubuntu/UnifiedQuantum-main/UnifiedQuantum-main')

import numpy as np
from uniqc.circuit_builder import Circuit
from uniqc.simulator import Simulator

NUM_QUBITS = 8
NUM_LAYERS = 8
NUM_CRZ_PER_LAYER = NUM_QUBITS - 1
NUM_RY_PER_LAYER = NUM_QUBITS
NUM_PARAMS_PER_LAYER = NUM_CRZ_PER_LAYER + NUM_RY_PER_LAYER
NUM_PARAMS = NUM_PARAMS_PER_LAYER * NUM_LAYERS

# 使用固定数据
np.random.seed(42)
encoded_data = np.random.uniform(-1, 1, NUM_QUBITS)
target = 1  # 标签


def build_circuit(encoded_data, params):
    circuit = Circuit(NUM_QUBITS)

    for layer in range(NUM_LAYERS):
        # RY(encoding)
        for i in range(NUM_QUBITS):
            value = encoded_data[i]
            angle = (np.pi / 16) * value
            circuit.ry(i, angle)

        # CRZ(theta)
        base_idx = layer * NUM_PARAMS_PER_LAYER
        crz_idx = base_idx
        for i in range(NUM_CRZ_PER_LAYER):
            theta = params[crz_idx + i]
            circuit.crz(i, i + 1, theta)

        # RY(theta)
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


def compute_loss(F, target):
    return (target - F) ** 2


def compute_gradient(encoded_data, target, params):
    """参数移位法则计算梯度"""
    grad = np.zeros(NUM_PARAMS)
    shift = np.pi / 2

    F_current = run_circuit(encoded_data, params)
    coefficient = -2 * (F_current - target)

    for i in range(NUM_PARAMS):
        params_plus = params.copy()
        params_plus[i] += shift
        F_plus = run_circuit(encoded_data, params_plus)

        params_minus = params.copy()
        params_minus[i] -= shift
        F_minus = run_circuit(encoded_data, params_minus)

        grad[i] = coefficient * (F_plus - F_minus) / 2

    return grad, F_current


def compute_numerical_gradient(encoded_data, target, params, epsilon=1e-5):
    """数值微分计算梯度（验证用）"""
    grad = np.zeros(NUM_PARAMS)

    for i in range(NUM_PARAMS):
        params_plus = params.copy()
        params_plus[i] += epsilon
        F_plus = run_circuit(encoded_data, params_plus)

        params_minus = params.copy()
        params_minus[i] -= epsilon
        F_minus = run_circuit(encoded_data, params_minus)

        loss_plus = compute_loss(F_plus, target)
        loss_minus = compute_loss(F_minus, target)

        grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

    return grad


# 测试
print("=" * 60)
print("梯度验证测试")
print("=" * 60)

# 初始化参数
np.random.seed(42)
params = np.random.uniform(-np.pi, np.pi, NUM_PARAMS)

print(f"\n初始参数范围: [{params.min():.4f}, {params.max():.4f}]")

# 计算解析梯度
grad_analytic, F_current = compute_gradient(encoded_data, target, params)
loss = compute_loss(F_current, target)

print(f"\n当前输出 F: {F_current:.6f}")
print(f"当前损失 L: {loss:.6f}")

# 计算数值梯度（很慢，只算前5个）
print("\n前5个参数的梯度对比:")
print("-" * 60)
print(f"{'参数i':<10}{'解析梯度':<20}{'数值梯度':<20}{'差异':<15}")
print("-" * 60)

for i in range(5):
    # 临时改参数
    orig = params[i]
    eps = 1e-5

    params_plus = params.copy()
    params_plus[i] += eps
    F_plus = run_circuit(encoded_data, params_plus)
    loss_plus = compute_loss(F_plus, target)

    params_minus = params.copy()
    params_minus[i] -= eps
    F_minus = run_circuit(encoded_data, params_minus)
    loss_minus = compute_loss(F_minus, target)

    grad_numeric = (loss_plus - loss_minus) / (2 * eps)

    diff = abs(grad_analytic[i] - grad_numeric)

    print(f"{i:<10}{grad_analytic[i]:<20.8f}{grad_numeric:<20.8f}{diff:<15.8f}")

    params[i] = orig

print("-" * 60)

# 检查梯度是否有效
grad_norm = np.linalg.norm(grad_analytic)
print(f"\n解析梯度范数: {grad_norm:.8f}")

if grad_norm < 1e-10:
    print("⚠️ 梯度几乎为零，可能有问题！")
else:
    print("✓ 梯度不为零")

# 检查几个样本的损失和梯度方向
print("\n" + "=" * 60)
print("梯度方向测试")
print("=" * 60)

# 用小学习率更新，看损失是否下降
lr = 0.01
params_test = params.copy()
loss_before = compute_loss(run_circuit(encoded_data, params_test), target)

grad_test, _ = compute_gradient(encoded_data, target, params_test)
params_test = params_test - lr * grad_test

loss_after = compute_loss(run_circuit(encoded_data, params_test), target)

print(f"学习率: {lr}")
print(f"更新前损失: {loss_before:.8f}")
print(f"更新后损失: {loss_after:.8f}")
print(f"损失变化: {loss_after - loss_before:.8f}")

if loss_after < loss_before:
    print("✓ 损失下降了！")
else:
    print("❌ 损失上升了！")
