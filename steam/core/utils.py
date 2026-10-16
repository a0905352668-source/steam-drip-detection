#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/utils.py - 通用工具函数
"""

import random
import numpy as np
import torch
from typing import Tuple


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_3floats(s: str) -> Tuple[float, float, float]:
    """
    解析逗号分隔的三个浮点数
    例如: '0.5,0.5,0.5' -> (0.5, 0.5, 0.5)
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("需要三个逗号分隔的数值，例如: '0.5,0.5,0.5'")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise ValueError("mean/std 必须是数值，例如: '0.5,0.5,0.5'")
    return vals


class EMA:
    """指数滑动平均 (Exponential Moving Average)"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def compute_metrics(preds, labels, num_classes=2):
    """
    计算分类指标

    Returns:
        dict: 包含 accuracy, precision, recall, f1
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    preds = np.array(preds)
    labels = np.array(labels)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='binary', zero_division=0),
        "recall": recall_score(labels, preds, average='binary', zero_division=0),
        "f1": f1_score(labels, preds, average='binary', zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist()
    }

    return metrics
