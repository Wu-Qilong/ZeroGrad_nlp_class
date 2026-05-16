import random
import numpy as np
import torch

from llm import SimpleLLM
from solver import TextSpaceZOSudokuSolver

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PUZZLE_4X4 = [
    [1, 0, 4, 0],
    [0, 3, 0, 2],
    [3, 0, 2, 0],
    [0, 4, 0, 1],
]

N_ITERATIONS = 20  
MU           = 0.05  
LR           = 0.03  


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

if __name__ == "__main__":
    main()
