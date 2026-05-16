from typing import List

def score_board(board: List[List[int]], size: int) -> float:
    loss = 0.0
    box_size = int(size ** 0.5)

    for i in range(size):
        for num in range(1, size + 1):
            c = board[i].count(num)
            if c > 1:
                loss += (c - 1) * 10

    for j in range(size):
        col = [board[i][j] for i in range(size)]
        for num in range(1, size + 1):
            c = col.count(num)
            if c > 1:
                loss += (c - 1) * 10

    for bi in range(box_size):
        for bj in range(box_size):
            box = [
                board[i][j]
                for i in range(bi * box_size, (bi + 1) * box_size)
                for j in range(bj * box_size, (bj + 1) * box_size)
            ]
            for num in range(1, size + 1):
                c = box.count(num)
                if c > 1:
                    loss += (c - 1) * 10

    for i in range(size):
        for j in range(size):
            val = board[i][j]
            if val == 0:
                loss += 2
            elif val < 1 or val > size:
                loss += 20

    return loss


def is_valid_solution(board: List[List[int]], size: int) -> bool:
    if any(0 in row for row in board):
        return False
    expected = list(range(1, size + 1))
    box_size = int(size ** 0.5)

    for i in range(size):
        if sorted(board[i]) != expected:
            return False
        col = [board[r][i] for r in range(size)]
        if sorted(col) != expected:
            return False

    for bi in range(box_size):
        for bj in range(box_size):
            box = [
                board[i][j]
                for i in range(bi * box_size, (bi + 1) * box_size)
                for j in range(bj * box_size, (bj + 1) * box_size)
            ]
            if sorted(box) != expected:
                return False

    return True
