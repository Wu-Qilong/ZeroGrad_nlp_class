"""
trainer_4x4.py — 4x4 数独训练入口

数字范围：1–4
每行、每列、每个 2x2 宫格中数字各出现一次。

用法：
    python trainer_4x4.py
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

# ── 4x4 谜题（0 表示空格）────────────────────────────────────────────────
PUZZLE_4X4 = [
    [1, 0, 4, 0],
    [0, 3, 0, 2],
    [3, 0, 2, 0],
    [0, 4, 0, 1],
]

# ── 超参数 ────────────────────────────────────────────────────────────────
N_ITERATIONS = 20   # 迭代次数（4x4 较简单，轮次少）
MU           = 0.05  # 扰动幅度
LR           = 0.03  # 学习率


def main():
    print("=" * 60)
    print("4x4 SUDOKU — Text-Space ZeroGrad Optimizer")
    print("=" * 60)

    llm    = SimpleLLM()
    solver = TextSpaceZOSudokuSolver(llm)

    result = solver.solve(
        puzzle       = PUZZLE_4X4,
        n_iterations = N_ITERATIONS,
        mu           = MU,
        lr           = LR,
        verbose      = True,
    )

    if result:
        print("\n" + "🎉" * 20)
        print("4x4 Sudoku solved successfully!")
        print("🎉" * 20)
    else:
        print("\n⚠️  Could not find a perfect solution within the iteration budget.")
        print("    Try increasing N_ITERATIONS or adjusting MU / LR.")


if __name__ == "__main__":
    main()
