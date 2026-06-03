"""
数据加载与预处理模块
"""
import numpy as np


def load_data():
    """
    加载训练集和测试集

    Returns:
        x_train: (1000, 256) 训练数据
        y_train: (1000,) 训练标签
        x_test: (200, 256) 测试数据
        y_test: (200,) 测试标签
    """
    # 加载训练数据
    train_data = np.load("/home/ubuntu/mnist_train_1000_16_16.npz")
    x_train = train_data['data']  # (1000, 16, 16)
    y_train = train_data['label']  # (1000,)

    # 加载测试数据
    test_data = np.load("/home/ubuntu/mnist_test_200_16_16.npz")
    x_test = test_data['data']  # (200, 16, 16)
    y_test = test_data['label']  # (200,)

    return x_train, y_train, x_test, y_test


def preprocess(x):
    """
    数据预处理：展平 + L2归一化

    Args:
        x: (N, 16, 16) 或 (N, 256) 图片数据

    Returns:
        x: (N, 256) 展平并归一化后的数据
    """
    # 展平为 256 维向量
    if len(x.shape) == 3:
        x = x.reshape(x.shape[0], -1)  # (N, 256)

    # L2 归一化（每个样本的模长变为 1）
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # 避免除零
    norms = np.where(norms == 0, 1, norms)
    x = x / norms

    return x


def prepare_data():
    """
    加载并预处理所有数据

    Returns:
        x_train, y_train, x_test, y_test
    """
    x_train, y_train, x_test, y_test = load_data()
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    return x_train, y_train, x_test, y_test
