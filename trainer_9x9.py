"""
trainer_9x9.py — 9x9 数独训练入口

数字范围：1–9
每行、每列、每个 3x3 宫格中数字各出现一次。

用法：
    python trainer_9x9.py
"""

import random
import numpy as np
import torch

from llm import SimpleLLM
from solver import TextSpaceZOSudokuSolver

# ── 固定随机种子，保证可复现 ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 9x9 谜题（0 表示空格）────────────────────────────────────────────────
PUZZLE_9X9 = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

# ── 超参数 ────────────────────────────────────────────────────────────────
N_ITERATIONS = 30   # 迭代次数（9x9 搜索空间更大，轮次更多）
MU           = 0.08  # 扰动幅度
LR           = 0.04  # 学习率


def main():
    print("=" * 60)
    print("9x9 SUDOKU — Text-Space ZeroGrad Optimizer")
    print("=" * 60)

    llm    = SimpleLLM()
    solver = TextSpaceZOSudokuSolver(llm)

    result = solver.solve(
        puzzle       = PUZZLE_9X9,
        n_iterations = N_ITERATIONS,
        mu           = MU,
        lr           = LR,
        verbose      = True,
    )

    if result:
        print("\n" + "🎉" * 20)
        print("9x9 Sudoku solved successfully!")
        print("🎉" * 20)
    else:
        print("\n⚠️  Could not find a perfect solution within the iteration budget.")
        print("    Try increasing N_ITERATIONS or adjusting MU / LR.")


if __name__ == "__main__":
    main()
