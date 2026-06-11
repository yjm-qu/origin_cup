"""
MNIST二分类 严格按6.10文档实现
8量子比特复用/63趟/Phase1固定+Phase2可训练
"""
import os
os.environ['PYTHONUNBUFFERED'] = '1'
import numpy as np
import time
import torch
import torch.nn as nn
import torchquantum as tq
from uniqc.circuit_builder import Circuit
from uniqc.simulator.torchquantum_simulator import _GATE_MAP, _DAGGER_MAP

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 超参数
EPOCHS = 3
NUM_TRAIN = 50
NUM_TEST = 200
LR = 0.01
D = 10  # Phase2 ansatz层数

torch.manual_seed(42)
np.random.seed(42)

# ============= 量子电路执行器 =============
def execute_circuit(opcode_list, param_overrides, n_qubits, init_state):
    qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=1, device="cpu")
    dim = 2 ** n_qubits
    sv = init_state.clone().to(torch.cfloat).cpu()
    qdev.states = sv.reshape(1, *[2] * n_qubits)

    for op_idx, opcode in enumerate(opcode_list):
        op_name, qubits, _cbits, params, dagger, controls = opcode
        if dagger and op_name in _DAGGER_MAP:
            gate_fn, is_parametric = _DAGGER_MAP[op_name]
        elif op_name in _GATE_MAP:
            gate_fn, is_parametric = _GATE_MAP[op_name]
        else:
            continue

        wires = qubits if isinstance(qubits, list) else [qubits]
        if controls:
            wires = list(controls) + wires

        if op_idx in param_overrides:
            gate_params = param_overrides[op_idx]
        elif is_parametric and params is not None:
            raw = [params] if not isinstance(params, (list, tuple)) else list(params)
            gate_params = torch.tensor(raw, dtype=torch.float32, device="cpu")
        else:
            gate_params = None

        kwargs = {"wires": wires, "inverse": False}
        if gate_params is not None:
            if gate_params.dim() == 0:
                gate_params = gate_params.unsqueeze(0).unsqueeze(0)
            elif gate_params.dim() == 1:
                gate_params = gate_params.unsqueeze(0)
            kwargs["params"] = gate_params
        gate_fn(qdev, **kwargs)

    states = qdev.get_states_1d().squeeze(0)
    return states

# ============= Phase1一趟（固定ansatz，零参数）============
def phase1_one_trip(pixels, trip_idx, reuse_sv):
    """
    Phase1的一趟
    trip_idx: 0-62
    reuse_sv: 复用组的量子态（4 qubit）
    """
    n_qubits = 8
    reuse_qubits = [0, 1, 2, 3]  # 复用组
    output_qubits = [4, 5, 6, 7]  # 输出组

    circuit = Circuit(n_qubits)

    # ① 角度编码 Ry(x)
    if trip_idx == 0:
        # 第0趟：装x0-x7（8像素）
        for i in range(8):
            circuit.ry(i, np.pi * pixels[i])
    else:
        # 第1-62趟：装4像素到复用组
        pixel_start = 8 + (trip_idx - 1) * 4
        for i in range(4):
            circuit.ry(reuse_qubits[i], np.pi * pixels[pixel_start + i])

    # ② 复用组固定ansatz: Ry(π/4)
    for q in reuse_qubits:
        circuit.ry(q, np.pi / 4)

    # ③ 复用组内CZ链: q0-q1, q1-q2, q2-q3
    for i in range(3):
        circuit.cz(reuse_qubits[i], reuse_qubits[i+1])

    # ④ 输出组固定ansatz: Ry(π/4)
    for q in output_qubits:
        circuit.ry(q, np.pi / 4)

    # ⑤ 输出组内CZ链: q4-q5, q5-q6, q6-q7
    for i in range(3):
        circuit.cz(output_qubits[i], output_qubits[i+1])

    # ⑥ 跨组CZ纠缠（单向）: q0→q4, q1→q5, q2→q6, q3→q7
    for i in range(4):
        circuit.cz(reuse_qubits[i], output_qubits[i])

    # ⑥.5 获取当前状态向量用于测量
    temp_sv = execute_circuit(circuit.opcode_list, {}, n_qubits, reuse_sv)

    # ⑦ 测量复用组（注释掉，测试梯度）
    # measure_results = []
    # for q in range(4):
    #     # 计算|0⟩和|1⟩的概率
    #     prob0 = sum(abs(temp_sv[i].item())**2 for i in range(len(temp_sv)) if (i >> q) & 1 == 0)
    #     measure_results.append(1 if prob0 < 0.5 else 0)

    # ⑧ 根据测量结果进行 reset（注释掉，测试梯度）
    # reset_sv = temp_sv.clone()
    # for q_idx, result in enumerate(measure_results):
    #     if result == 1:
    #         for i in range(len(reset_sv)):
    #             if (i >> q_idx) & 1 == 1:
    #                 j = i ^ (1 << q_idx)
    #                 reset_sv[i], reset_sv[j] = reset_sv[j], reset_sv[i]

    # ⑨ 执行电路（直接用temp_sv，不reset）
    new_sv = temp_sv

    # 返回输出组状态
    return new_sv

# ============= Phase2（D层可训练ansatz）============
def phase2(params):
    """
    Phase2: D层 × (4个Rx + 3个CZ)
    params: 4D个参数
    返回: circuit, param_overrides
    """
    circuit = Circuit(8)  # 8个量子位: q0-q7
    param_overrides = {}

    param_idx = 0

    # D层
    for layer in range(D):
        # 4个Rx(θ)门
        for q in range(4, 8):  # 作用在输出组 q4-q7
            circuit.rx(q, 0.0)  # 参数由外部控制
            param_overrides[len(circuit.opcode_list) - 1] = params[param_idx:param_idx+1].cpu()
            param_idx += 1

        # 3个CZ门: q4-q5, q5-q6, q6-q7
        for i in range(3):
            circuit.cz(i, i+1)

    return circuit, param_overrides

# ============= 完整前向传播 =============
def forward_one_sample(pixels, params):
    """
    单个样本的完整前向传播
    严格按6.10文档实现
    pixels: 256维
    params: 4D个参数
    """
    n_qubits = 8

    # 初始态: |00000000⟩
    init_state = torch.zeros(2**n_qubits, dtype=torch.cfloat, device="cpu")
    init_state[0] = 1.0

    # Phase 1: 63趟
    current_sv = init_state

    # 第0趟
    current_sv = phase1_one_trip(pixels, 0, current_sv)

    # 第1-62趟
    for trip_idx in range(1, 63):
        current_sv = phase1_one_trip(pixels, trip_idx, current_sv)

    # Phase 2: 可训练ansatz
    phase2_circuit, phase2_params = phase2(params)

    # CNOT链汇聚 q4→q5→q6→q7
    phase2_circuit.cnot(4, 5)
    phase2_circuit.cnot(5, 6)
    phase2_circuit.cnot(6, 7)

    # 执行 Phase 2 (带 CNOT 链)
    final_sv = execute_circuit(
        phase2_circuit.opcode_list,
        phase2_params,
        8,
        current_sv
    )

    # 测q7（对应输出组的q3）
    prob1 = torch.abs(final_sv[1].clone()) ** 2  # |00000001⟩ = q7=1
    prob1 = torch.clamp(prob1, 1e-7, 1 - 1e-7)

    # logit = ln(p/(1-p))
    logit = torch.log(prob1 / (1 - prob1))

    return logit

# ============= 模型 =============
class QuantumModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 4D个参数（Phase2）
        self.params = nn.Parameter(torch.randn(4 * D) * 0.1)

    def forward(self, x):
        """
        x: (batch, 16, 16) -> (batch, 1) logits
        """
        batch_size = x.shape[0]

        # 编码: 16x16 -> 256维
        pixels = x.reshape(batch_size, 256)

        outputs = []
        for i in range(batch_size):
            logit = forward_one_sample(pixels[i], self.params)
            outputs.append(logit)

        return torch.stack(outputs).unsqueeze(1)

# ============= 主流程 =============
def main():
    print("="*60)
    print("MNIST二分类 严格按6.10文档实现")
    print("="*60)

    # 加载数据
    train_npz = np.load(os.path.join(DATA_DIR, '../整理文件/mnist_train_1000_16_16.npz'))
    test_npz = np.load(os.path.join(DATA_DIR, '../整理文件/mnist_test_200_16_16.npz'))

    x_train = train_npz['data'][:NUM_TRAIN].astype(np.float32)
    y_train = train_npz['label'][:NUM_TRAIN].astype(np.float32)
    x_test = test_npz['data'][:NUM_TEST].astype(np.float32)
    y_test = test_npz['label'][:NUM_TEST].astype(np.float32)

    # 标签映射
    label_map = {float(v): i for i, v in enumerate(np.unique(y_train))}
    y_train = np.array([label_map[float(y)] for y in y_train], dtype=np.float32)
    y_test = np.array([label_map[float(y)] for y in y_test], dtype=np.float32)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    model = QuantumModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print(f"训练样本: {NUM_TRAIN}, 测试样本: {NUM_TEST}")
    print(f"量子参数: {model.params.numel()}")
    print()

    with open("train_log.txt", "w") as f:
        f.write("epoch\tepoch_time\tloss\ttest_acc\n")

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()

        # 训练每个样本
        for sample_idx in range(NUM_TRAIN):
            sample_start = time.time()

            optimizer.zero_grad()
            output = model(x_train_t[sample_idx:sample_idx+1])
            loss = criterion(output, y_train_t[sample_idx:sample_idx+1])
            loss.backward()
            optimizer.step()

            sample_time = time.time() - sample_start
            print(f"[样本{sample_idx+1}/{NUM_TRAIN}] 损失: {loss.item():.6f}, 时间: {sample_time:.3f}秒", flush=True)

        epoch_time = time.time() - epoch_start

        # 计算平均损失
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i in range(NUM_TRAIN):
                output = model(x_train_t[i:i+1])
                loss = criterion(output, y_train_t[i:i+1])
                total_loss += loss.item()
            avg_loss = total_loss / NUM_TRAIN

            # 测试准确率
            preds = (torch.sigmoid(model(x_test_t)) > 0.5).float()
            correct = (preds.squeeze() == y_test_t).sum().item()
            acc = correct / NUM_TEST * 100

        print(f"Epoch {epoch+1}/{EPOCHS} 完成 - 损失: {avg_loss:.6f}, 时间: {epoch_time:.1f}秒", flush=True)
        print(f"测试准确率: {correct}/{NUM_TEST}={acc:.1f}%", flush=True)

        with open("train_log.txt", "a") as f:
            f.write(f"{epoch+1}\t{epoch_time:.2f}\t{avg_loss:.6f}\t{acc:.2f}\n")

    total_time = time.time() - start_time
    print(f"\n总训练时间: {total_time:.0f}秒")

if __name__ == "__main__":
    main()
