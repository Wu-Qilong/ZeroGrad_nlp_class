"""
parser.py — 文本解析 & 棋盘打印工具
"""

import re
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
#  解析
# ─────────────────────────────────────────────────────────────────────────────

def board_to_str(board: List[List[int]]) -> str:
    """将棋盘转为字符串（空格用 '.' 表示）。"""
    return '\n'.join(
        ' '.join(str(c) if c != 0 else '.' for c in row)
        for row in board
    )


def parse_board_from_text(
    text: str,
    size: int,
    original_board: List[List[int]],
) -> List[List[int]]:
    """
    从 LLM 输出中解析 N×N 数独棋盘。

    支持两种格式：
      1. (row,col): val  逐格格式
      2. 每行一行数字（空格/逗号分隔）的 grid 格式

    解析失败的格子：
      - 已知格 → 保留 original_board 值
      - 空格   → 填 0
    """
    board = [row[:] for row in original_board]

    # 格式 1：(row,col): val
    pattern_cell = r'\((\d+)\s*,\s*(\d+)\)\s*[:\-=]\s*(\d+)'
    matches = re.findall(pattern_cell, text)
    if matches:
        for r, c, v in matches:
            ri, ci, vi = int(r), int(c), int(v)
            if 0 <= ri < size and 0 <= ci < size and 1 <= vi <= size:
                if original_board[ri][ci] == 0:
                    board[ri][ci] = vi
        return board

    # 格式 2：grid 逐行
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


# ─────────────────────────────────────────────────────────────────────────────
#  打印
# ─────────────────────────────────────────────────────────────────────────────

def print_board(board: List[List[int]]) -> None:
    """带宫格分隔线的棋盘打印。"""
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
    """以 Python list 格式打印棋盘。"""
    print("[")
    for row in board:
        print(f"  {row},")
    print("]")
