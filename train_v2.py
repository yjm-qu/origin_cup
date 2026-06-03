"""
量子分类器训练脚本（参数化量子线路版本）
数值梯度训练
"""
import sys
sys.path.insert(0, '/home/ubuntu/UnifiedQuantum-main/UnifiedQuantum-main')

# 直接运行
sys.path.insert(0, '/home/ubuntu/origin_cup')
from utils.circuit_v2 import train, test

if __name__ == "__main__":
    params = train()
    test(params)
