"""
量子分类器训练脚本
方案：量子特征提取 + 经典分类器
"""
import sys
sys.path.insert(0, '/home/ubuntu/UnifiedQuantum-main/UnifiedQuantum-main')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils.data import prepare_data
from utils.circuit import extract_features_batch, NUM_QUBITS


# 超参数
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32


class QuantumClassifier(nn.Module):
    """
    经典分类器（接收量子特征作为输入）
    """
    def __init__(self, input_size=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


def train():
    print("Loading data...")
    x_train, y_train, x_test, y_test = prepare_data()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    # 提取量子特征（离线）
    print("Extracting quantum features (training)...")
    x_train_features = extract_features_batch(x_train)
    print(f"Quantum features shape: {x_train_features.shape}")

    # 转换数据
    x_train_tensor = torch.tensor(x_train_features, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    # 数据加载器
    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型
    model = QuantumClassifier(input_size=x_train_features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # 训练
    print(f"Training for {EPOCHS} epochs...")
    best_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    print(f"Training complete! Best loss: {best_loss:.4f}")

    # 保存模型
    torch.save(best_state, "/home/ubuntu/origin_cup/model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    train()
