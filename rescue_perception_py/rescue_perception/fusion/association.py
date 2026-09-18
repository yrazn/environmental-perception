"""目标关联工具：匈牙利最小代价分配、代价门控和二维 IoU。"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """求矩形代价矩阵的全局最小一对一分配。"""
    """Minimum-cost matching. Returns list of (row, col) pairs."""
    if cost.size == 0:
        return []
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    transposed = n > m
    if transposed:
        cost = cost.T
        n, m = cost.shape

    big = float(np.finfo(float).max) / 4.0
    a = np.asarray(cost, dtype=float).copy()
    a[np.isnan(a)] = big
    a[a > big] = big

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, big)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = big
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = [(int(p[j]) - 1, int(j) - 1) for j in range(1, m + 1) if p[j] != 0]
    if transposed:
        pairs = [(j, i) for i, j in pairs]
    return pairs


def gated_hungarian(cost: np.ndarray, max_cost: float) -> List[Tuple[int, int]]:
    """先执行全局分配，再剔除超过物理合理门限的匹配。"""
    pairs = hungarian(cost)
    return [(i, j) for i, j in pairs if float(cost[i, j]) <= max_cost]


def iou_2d(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU for axis-aligned boxes [x1, y1, x2, y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
