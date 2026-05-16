import numpy as np
from typing import List, Optional, Tuple

from llm import SimpleLLM
from scorer import score_board, is_valid_solution
from parser import board_to_str, parse_board_from_text, print_board, print_array

class TextSpaceZOSudokuSolver:

    def __init__(self, llm: SimpleLLM):
        self.llm = llm
        self.optimization_history: List[dict] = []

    def _initial_prompt(self, puzzle: List[List[int]], size: int) -> str:
        puzzle_str = board_to_str(puzzle)
        prompt = f"""You are an expert Sudoku solver.
Below is a {size}x{size} Sudoku puzzle. '.' means an empty cell.
Numbers must be 1–{size}. Each row, column, and {int(size**0.5)}x{int(size**0.5)} box must contain each number exactly once.

Puzzle:
{puzzle_str}

Please provide a complete solution as a {size}x{size} grid.
Each row on its own line, numbers separated by spaces.
Only output the grid, nothing else."""
        return self.llm.generate(prompt, max_tokens=300)

    def _project_vector_to_text(
        self,
        base_vec: np.ndarray,
        perturbed_vec: np.ndarray,
        current_text: str,
        puzzle: List[List[int]],
        size: int,
        direction_label: str = "",
    ) -> str:
        diff = perturbed_vec - base_vec
        diff_norm = float(np.linalg.norm(diff))
        top_idx = np.argsort(np.abs(diff))[-8:]
        top_shifts = {int(i): round(float(diff[i]), 4) for i in top_idx}

        puzzle_str = board_to_str(puzzle)
        prompt = f"""You are an expert Sudoku solver doing iterative refinement.

Original puzzle ({size}x{size}, '.' = empty):
{puzzle_str}

Current solution attempt:
{current_text}

Semantic perturbation signal ({direction_label}):
  - Overall perturbation magnitude: {diff_norm:.4f}
  - Key dimension shifts (index: delta): {top_shifts}
  (Positive delta → explore bolder / more committed choices;
   Negative delta → explore safer / more conservative choices.)

Based on the perturbation signal, produce a refined {size}x{size} Sudoku solution.
If the perturbation is large-positive, try filling uncertain cells more aggressively.
If the perturbation is large-negative, be more conservative and revert questionable cells.

Output ONLY the {size}x{size} grid. Each row on its own line, numbers 1–{size} separated by spaces."""
        return self.llm.generate(prompt, max_tokens=300)

    def _project_gradient_update_to_text(
        self,
        base_vec: np.ndarray,
        updated_vec: np.ndarray,
        current_text: str,
        gradient_scalar: float,
        puzzle: List[List[int]],
        size: int,
    ) -> str:
        diff = updated_vec - base_vec
        diff_norm = float(np.linalg.norm(diff))
        puzzle_str = board_to_str(puzzle)

        direction_hint = (
            "Move toward the positive perturbation direction (fill more cells boldly)."
            if gradient_scalar > 0
            else "Move toward the negative perturbation direction (be more conservative, fix mistakes)."
        )

        prompt = f"""You are an expert Sudoku solver doing gradient-guided refinement.

Original puzzle ({size}x{size}, '.' = empty):
{puzzle_str}

Current solution attempt:
{current_text}

Gradient update signal:
  - Gradient scalar: {gradient_scalar:.4f}
  - Update vector norm: {diff_norm:.4f}
  - Direction: {direction_hint}

Apply this gradient update: produce an improved {size}x{size} Sudoku solution that moves
in the indicated direction. Fix the most error-prone cells first.

Output ONLY the {size}x{size} grid. Each row on its own line, numbers 1–{size} separated by spaces."""
        return self.llm.generate(prompt, max_tokens=300)

    def optimize(
        self,
        puzzle: List[List[int]],
        n_iterations: int = 30,
        mu: float = 0.1,
        lr: float = 0.05,
        verbose: bool = True,
    ) -> Tuple[str, List[List[int]]]:
        size = len(puzzle)
        original_board = [row[:] for row in puzzle]

        def log(msg: str):
            if verbose:
                print(msg)

        current_text = self._initial_prompt(puzzle, size)

        current_board = parse_board_from_text(current_text, size, original_board)
        current_score = score_board(current_board, size)

        best_score = current_score
        best_text = current_text
        best_board = [row[:] for row in current_board]
        self.optimization_history = []

        for iteration in range(n_iterations):
            base_vec = self.llm.embed(current_text)
            d = base_vec.shape[0]

            u = np.random.randn(d)
            u = u / (np.linalg.norm(u) + 1e-12)

            vec_plus  = base_vec + mu * u
            vec_minus = base_vec - mu * u

            text_plus = self._project_vector_to_text(
                base_vec, vec_plus, current_text, puzzle, size, direction_label="+μu"
            )
            text_minus = self._project_vector_to_text(
                base_vec, vec_minus, current_text, puzzle, size, direction_label="-μu"
            )

            board_plus  = parse_board_from_text(text_plus,  size, original_board)
            board_minus = parse_board_from_text(text_minus, size, original_board)
            score_plus  = score_board(board_plus,  size)
            score_minus = score_board(board_minus, size)

            grad_scalar = (score_plus - score_minus) / (2.0 * mu)
            updated_vec = base_vec - lr * grad_scalar * u

            new_text = self._project_gradient_update_to_text(
                base_vec, updated_vec, current_text, grad_scalar, puzzle, size
            )

            new_board = parse_board_from_text(new_text, size, original_board)
            new_score = score_board(new_board, size)

            self.optimization_history.append({
                'iteration': iteration,
                'score': new_score,
                'grad_scalar': grad_scalar,
            })

            if new_score < current_score:
                improvement = current_score - new_score
                current_text  = new_text
                current_board = new_board
                current_score = new_score

                if new_score < best_score:
                    best_score  = new_score
                    best_text   = new_text
                    best_board  = [row[:] for row in new_board]
                  
            if best_score == 0:
                log("\n🎉 Perfect solution found! Stopping early.")
                break

        return best_text, best_board

    def solve(
        self,
        puzzle: List[List[int]],
        n_iterations: int = 30,
        mu: float = 0.1,
        lr: float = 0.05,
        verbose: bool = True,
    ) -> Optional[List[List[int]]]:
        size = len(puzzle)

        print("\n" + "="*60)
        print(f"{size}x{size} SUDOKU — Text-Space ZeroGrad Optimizer")
        print("="*60)
        print("\n【Initial Puzzle】")
        print_board(puzzle)

        best_text, best_board = self.optimize(puzzle, n_iterations, mu, lr, verbose)

        print("\n" + "="*60)
        print("【FINAL RESULT】")
        print("="*60)
        print(f"\nBest solution text:\n{best_text}\n")
