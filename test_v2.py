"""
量子分类器测试脚本（参数化量子线路版本）
"""
import sys
sys.path.insert(0, '/home/ubuntu/UnifiedQuantum-main/UnifiedQuantum-main')

import numpy as np
import torch

# 导入数据处理
sys.path.insert(0, '/home/ubuntu/origin_cup')
from utils.data import prepare_data
from utils.circuit_v2 import QuantumClassifier, NUM_QUBITS


# 评委会替换的种子
RANDOM_SEED = 42


def test():
    print("Loading data...")
    x_train, y_train, x_test, y_test = prepare_data()
    print(f"Test: {x_test.shape}")

    # 加载模型
    print("Loading trained model...")
    model = QuantumClassifier()
    model.load_state_dict(torch.load("/home/ubuntu/origin_cup/model_v2.pth"))
    model.eval()

    # 测试
    print("Predicting...")
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    with torch.no_grad():
        predictions = model.predict(x_test_tensor)

    # 计算准确率
    accuracy = (predictions.numpy() == y_test).mean()
    print(f"Test Accuracy: {accuracy:.4f}")

    # 计算评分
    Q = NUM_QUBITS
    P = NUM_QUBITS * 5  # 40
    Acc = accuracy
    score = 44 * Acc + 3 * ((8 - Q) / 7 + (100 - P) / 99)

    # 输出
    print("\n=== Results ===")
    print("Score,Acc,Q,P")
    print(f"{score:.2f},{Acc:.2f},{Q},{P}")


if __name__ == "__main__":
    test()
