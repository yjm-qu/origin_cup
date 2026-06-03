"""
量子线路构建模块
使用 uniqc 框架进行特征提取 + DummyAdapter 噪声模拟
"""
import numpy as np
from uniqc.circuit_builder import Circuit
from uniqc.simulator import Simulator, NoisySimulator


# 量子比特数量
NUM_QUBITS = 8
NUM_LAYERS = 5


def build_encoding_circuit(data_vector):
    """
    构建编码电路
    """
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

    return circuit


def extract_quantum_features(data_vector):
    """
    提取量子特征（无噪声）
    """
    circuit = Circuit(NUM_QUBITS)

    # 编码层
    for i in range(NUM_QUBITS):
        start_idx = i * 32
        chunk = data_vector[start_idx:start_idx + 32]
        chunk_norm = np.linalg.norm(chunk)
        if chunk_norm > 1e-10:
            angle = np.arccos(np.clip(chunk[0] / chunk_norm, -1, 1))
        else:
            angle = 0
        circuit.ry(i, angle)

    # Ansatz 层
    for layer in range(NUM_LAYERS):
        for qubit in range(NUM_QUBITS):
            angle = (layer * NUM_QUBITS + qubit) * np.pi / 4
            circuit.ry(qubit, angle)
        for i in range(0, NUM_QUBITS - 1, 2):
            circuit.cnot(i, i + 1)
        for i in range(1, NUM_QUBITS - 1, 2):
            circuit.cnot(i, i + 1)

    # 测量所有量子比特
    for i in range(NUM_QUBITS):
        circuit.measure(i)

    # 模拟
    simulator = Simulator(backend_type="statevector")
    probs = simulator.simulate_pmeasure(circuit)

    # 转换为 dict
    prob_dict = {}
    for idx in range(len(probs)):
        bit_str = format(idx, f'0{NUM_QUBITS}b')
        prob_dict[bit_str] = probs[idx]

    # 提取特征
    features = []
    for i in range(NUM_QUBITS):
        prob_0 = 0.0
        for bit_str, val in prob_dict.items():
            if bit_str[NUM_QUBITS - 1 - i] == '0':
                prob_0 += val
        features.append(prob_0)

    return np.array(features[:16])


def extract_features_batch(data_matrix):
    """批量提取特征"""
    features = []
    for data in data_matrix:
        feat = extract_quantum_features(data)
        features.append(feat)
    return np.array(features)


def simulate_with_noise(data_vector, params=None, seed=42):
    """
    使用 DummyAdapter 噪声模拟

    Args:
        data_vector: (256,) 数据
        params: 参数（未使用）
        seed: 随机种子

    Returns:
        float: P(|0⟩) for qubit 0
    """
    from utils.noise import get_noise_model
    from uniqc.backend_adapter.task.adapters.dummy_adapter import DummyAdapter

    # 构建电路
    circuit = Circuit(NUM_QUBITS)

    # 编码
    for i in range(NUM_QUBITS):
        start_idx = i * 32
        chunk = data_vector[start_idx:start_idx + 32]
        chunk_norm = np.linalg.norm(chunk)
        if chunk_norm > 1e-10:
            angle = np.arccos(np.clip(chunk[0] / chunk_norm, -1, 1))
        else:
            angle = 0
        circuit.ry(i, angle)

    # Ansatz
    for layer in range(NUM_LAYERS):
        for qubit in range(NUM_QUBITS):
            angle = (layer * NUM_QUBITS + qubit) * np.pi / 4
            circuit.ry(qubit, angle)
        for i in range(0, NUM_QUBITS - 1, 2):
            circuit.cnot(i, i + 1)
        for i in range(1, NUM_QUBITS - 1, 2):
            circuit.cnot(i, i + 1)

    # 测量 qubit 0
    circuit.measure(0)

    # 获取噪声配置
    noise_model = get_noise_model(seed)

    # 使用 DummyAdapter
    adapter = DummyAdapter(noise_model=noise_model)

    # 提交任务
    task_id = adapter.submit(circuit.originir)

    # 获取结果
    result = adapter.query(task_id)

    # 解析结果 - 获取 P(|0⟩)
    if result.get('status') == 'success':
        # result 格式: {'00': 500, '11': 500}
        counts = result.get('result', {})
        total_shots = sum(counts.values())

        if total_shots > 0:
            # 计算 P(|0⟩) for qubit 0
            shots_0 = 0
            for bit_str, count in counts.items():
                # qubit 0 是最高位
                if len(bit_str) > 0 and bit_str[-1] == '0':
                    shots_0 += count
            prob_0 = shots_0 / total_shots
            return prob_0

    return 0.5  # 默认值
