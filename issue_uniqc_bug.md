# [Bug] TwoQubitDepolarizing 噪声模型导致 TypeError 错误

## Bug 描述

在 uniqc 库中使用 TwoQubitDepolarizing 噪声模型配合 NoisySimulator 的 density_matrix/density_operator 模式时，模拟会抛出 TypeError 错误：

```
TypeError: 'int' object is not subscriptable
```

## 复现步骤

```python
from uniqc.simulator.error_model import TwoQubitDepolarizing, ErrorLoader_GateSpecificError
from uniqc.circuit_builder import Circuit
from uniqc.simulator import NoisySimulator

# 创建去极化噪声
error = TwoQubitDepolarizing(0.0154)

# 配置对 CNOT 门添加噪声
gate_type_error = {'CNOT': [error]}
error_loader = ErrorLoader_GateSpecificError([], gate_type_error, {})

# 创建包含 CNOT 门的量子电路
circuit = Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)
circuit.measure(0)

# 使用噪声模拟器运行
noisy_sim = NoisySimulator(
    backend_type='density_operator',
    available_qubits=[0, 1],
    error_loader=error_loader
)
result = noisy_sim.simulate_pmeasure(circuit.originir)
```

## 预期行为

模拟应该正常运行，并在 CNOT 门上去极化噪声。

## 实际行为

抛出错误：`TypeError: 'int' object is not subscriptable`

## 受影响的门

| 门 | 状态 |
|------|--------|
| CNOT | ❌ 报错 |
| CZ | ❌ 报错 |
| XX (RXX) | ❌ 报错 |
| YY (RYY) | ❌ 报错 |
| ZZ (RZZ) | ❌ 报错 |
| CP | ✅ 正常（但可能未正确添加噪声） |

## 分析

问题出在 `TwoQubitDepolarizing.generate_error_opcode` 函数。该函数在生成噪声 opcode 时，将输入的列表格式 `qubits=[0,1]` 转换为两个独立的整数 opcode，qubit 字段分别是 0 和 1。但后续执行这些 opcode 时，代码假设 qubit 字段是列表格式，使用 `qubit[0]` 取值，导致整数无法用下标访问而报错。

## 环境

- uniqc 版本: 0.0.14
- Python 版本: 3.10
