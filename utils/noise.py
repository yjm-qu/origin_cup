"""
噪声模型配置模块
使用 uniqc 的 DummyAdapter 配置噪声
"""
import numpy as np


# 噪声基准参数（来自 noise.md）
BASE_RATE_2Q_ENT = 0.0154  # CNOT, CZ, CP
BASE_RATE_ANALOG = 0.0200   # RXX, RYY, RZZ, RZX
BASE_READOUT_ERROR = 0.0698  # 读取错误
PERTURBATION_SCALE = 0.05   # ±5% 扰动


def get_perturbed_rate(base_rate, rng):
    """
    根据基准错误率和随机数生成器，计算扰动后的错误率

    Args:
        base_rate: 基准错误率
        rng: numpy 随机数生成器

    Returns:
        扰动后的错误率
    """
    factor = rng.uniform(1 - PERTURBATION_SCALE, 1 + PERTURBATION_SCALE)
    perturbed_rate = base_rate * factor
    return np.clip(perturbed_rate, 0.0, 1.0)


def get_noise_model(seed=42):
    """
    根据随机种子生成噪声模型配置

    Args:
        seed: 随机种子

    Returns:
        dict: noise_model 配置字典
    """
    # 创建随机数生成器
    rng = np.random.default_rng(seed=seed)

    # 计算扰动后的错误率
    # 简化：使用双比特门错误率的平均值
    rate_2q = (BASE_RATE_2Q_ENT + BASE_RATE_ANALOG) / 2
    rate_2q = get_perturbed_rate(rate_2q, rng)

    read_error = get_perturbed_rate(BASE_READOUT_ERROR, rng)

    noise_model = {
        'depol_1q': 0.0,       # 单比特门无噪声
        'depol_2q': rate_2q,   # 双比特门去极化噪声
        'readout': read_error,  # 读取错误
    }

    return noise_model
