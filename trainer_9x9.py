import random
import numpy as np
import torch

from llm import SimpleLLM
from solver import TextSpaceZOSudokuSolver

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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

N_ITERATIONS = 30   
MU           = 0.08 
LR           = 0.04 


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

if __name__ == "__main__":
    main()
