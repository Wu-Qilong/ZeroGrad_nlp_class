import re
from typing import List

def board_to_str(board: List[List[int]]) -> str:
    return '\n'.join(
        ' '.join(str(c) if c != 0 else '.' for c in row)
        for row in board
    )

def parse_board_from_text(
    text: str,
    size: int,
    original_board: List[List[int]],
) -> List[List[int]]:
    board = [row[:] for row in original_board]

    pattern_cell = r'\((\d+)\s*,\s*(\d+)\)\s*[:\-=]\s*(\d+)'
    matches = re.findall(pattern_cell, text)
    if matches:
        for r, c, v in matches:
            ri, ci, vi = int(r), int(c), int(v)
            if 0 <= ri < size and 0 <= ci < size and 1 <= vi <= size:
                if original_board[ri][ci] == 0:
                    board[ri][ci] = vi
        return board

    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    number_lines = []
    for ln in lines:
        nums = re.findall(r'\b(\d)\b', ln)
        if len(nums) == size:
            number_lines.append([int(n) for n in nums])
        if len(number_lines) == size:
            break

    if len(number_lines) == size:
        for i in range(size):
            for j in range(size):
                if original_board[i][j] == 0 and 1 <= number_lines[i][j] <= size:
                    board[i][j] = number_lines[i][j]

    return board

def print_board(board: List[List[int]]) -> None:
    size = len(board)
    box_size = int(size ** 0.5)
    for i, row in enumerate(board):
        parts = []
        for j, c in enumerate(row):
            parts.append(str(c) if c != 0 else '.')
            if (j + 1) % box_size == 0 and j != size - 1:
                parts.append('|')
        print(' '.join(parts))
        if (i + 1) % box_size == 0 and i != size - 1:
            print('-' * (size * 2 + box_size - 1))
    print()


def print_array(board: List[List[int]]) -> None:
    print("[")
    for row in board:
        print(f"  {row},")
    print("]")
