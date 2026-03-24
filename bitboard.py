import numpy as np
import random

# Lookup Tables
# A row is represented by 16 bits: 4 tiles * 4 bits each.
# Total possible permutations = 2^16 = 65536.
ROW_LEFT_TABLE = np.zeros(65536, dtype=np.uint16)
ROW_RIGHT_TABLE = np.zeros(65536, dtype=np.uint16)
SCORE_TABLE = np.zeros(65536, dtype=np.uint32)
# Reversal table to help with Right moves easily if we compute Left first
REVERSE_ROW_TABLE = np.zeros(65536, dtype=np.uint16)

def init_tables():
    for row in range(65536):
        # Extract the 4 tiles
        line = [
            (row >> 12) & 0xF,
            (row >> 8) & 0xF,
            (row >> 4) & 0xF,
            (row >> 0) & 0xF
        ]
        
        # Calculate reverse
        rev_row = (line[3] << 12) | (line[2] << 8) | (line[1] << 4) | line[0]
        REVERSE_ROW_TABLE[row] = rev_row

        # Execute left move
        score = 0
        new_line = []
        
        # Filter out empty tiles
        non_empty = [t for t in line if t != 0]
        
        # Merge
        i = 0
        while i < len(non_empty):
            if i + 1 < len(non_empty) and non_empty[i] == non_empty[i+1] and non_empty[i] < 15:
                merged_val = non_empty[i] + 1
                new_line.append(merged_val)
                score += (1 << merged_val)  # 2^(val)
                i += 2
            else:
                new_line.append(non_empty[i])
                i += 1
                
        # Pad with empty
        while len(new_line) < 4:
            new_line.append(0)
            
        # Pack to 16 bits
        left_val = (new_line[0] << 12) | (new_line[1] << 8) | (new_line[2] << 4) | new_line[3]
        
        ROW_LEFT_TABLE[row] = left_val
        SCORE_TABLE[row] = score

    for row in range(65536):
        rev = REVERSE_ROW_TABLE[row]
        rev_left = ROW_LEFT_TABLE[rev]
        ROW_RIGHT_TABLE[row] = REVERSE_ROW_TABLE[rev_left]

# Pre-compute tables on load
init_tables()

# ---------------------------------------------------------
# Bitboard Board representation
# 16 tiles flattened, labeled 0 to 15.
#
#  0  1  2  3  -> Row 0 (Shift 48 to 63)
#  4  5  6  7  -> Row 1 (Shift 32 to 47)
#  8  9 10 11  -> Row 2 (Shift 16 to 31)
# 12 13 14 15  -> Row 3 (Shift 0 to 15)
# ---------------------------------------------------------

ROW_MASKS = [
    0xFFFF000000000000,
    0x0000FFFF00000000,
    0x00000000FFFF0000,
    0x000000000000FFFF
]

COL_MASK = 0x000F000F000F000F

def transpose(board: int) -> int:
    """Transpose the 64-bit board"""
    res = 0
    for i in range(16):
        r = i // 4
        c = i % 4
        j = c * 4 + r
        val = (board >> (i * 4)) & 0xF
        res |= (val << (j * 4))
    return res

def move_left(board: int):
    new_board = 0
    score = 0
    t0 = (board >> 48) & 0xFFFF
    t1 = (board >> 32) & 0xFFFF
    t2 = (board >> 16) & 0xFFFF
    t3 = board & 0xFFFF
    
    new_board |= int(ROW_LEFT_TABLE[t0]) << 48
    new_board |= int(ROW_LEFT_TABLE[t1]) << 32
    new_board |= int(ROW_LEFT_TABLE[t2]) << 16
    new_board |= int(ROW_LEFT_TABLE[t3])
    
    score += int(SCORE_TABLE[t0])
    score += int(SCORE_TABLE[t1])
    score += int(SCORE_TABLE[t2])
    score += int(SCORE_TABLE[t3])
    
    return new_board, score

def move_right(board: int):
    new_board = 0
    score = 0
    t0 = (board >> 48) & 0xFFFF
    t1 = (board >> 32) & 0xFFFF
    t2 = (board >> 16) & 0xFFFF
    t3 = board & 0xFFFF
    
    new_board |= int(ROW_RIGHT_TABLE[t0]) << 48
    new_board |= int(ROW_RIGHT_TABLE[t1]) << 32
    new_board |= int(ROW_RIGHT_TABLE[t2]) << 16
    new_board |= int(ROW_RIGHT_TABLE[t3])
    
    score += int(SCORE_TABLE[t0])
    score += int(SCORE_TABLE[t1])
    score += int(SCORE_TABLE[t2])
    score += int(SCORE_TABLE[t3])
    
    return new_board, score

def move_up(board: int):
    t_board = transpose(board)
    new_board, score = move_left(t_board)
    return transpose(new_board), score

def move_down(board: int):
    t_board = transpose(board)
    new_board, score = move_right(t_board)
    return transpose(new_board), score

def count_empty(board: int) -> int:
    empty = 0
    for i in range(16):
        if ((board >> (i * 4)) & 0xF) == 0:
            empty += 1
    return empty

def get_empty_positions(board: int):
    empty_pos = []
    for i in range(16):
        if ((board >> (i * 4)) & 0xF) == 0:
            empty_pos.append(i)
    return empty_pos

def insert_random_tile(board: int) -> int:
    empty_pos = get_empty_positions(board)
    if not empty_pos:
        return board
    pos = random.choice(empty_pos)
    # 90% chance of 2 (val=1), 10% chance of 4 (val=2)
    val = 1 if random.random() < 0.9 else 2
    return board | (val << (pos * 4))

def is_terminal(board: int) -> bool:
    if count_empty(board) > 0:
        return False
    # Check if any move changes the board
    b_l, _ = move_left(board)
    if b_l != board: return False
    b_r, _ = move_right(board)
    if b_r != board: return False
    b_u, _ = move_up(board)
    if b_u != board: return False
    b_d, _ = move_down(board)
    if b_d != board: return False
    return True

def print_board(board: int):
    for r in range(4):
        row = (board >> ((3 - r) * 16)) & 0xFFFF
        tiles = [(row >> 12) & 0xF, (row >> 8) & 0xF, (row >> 4) & 0xF, row & 0xF]
        print(" ".join(f"{(1 << t) if t > 0 else 0:4d}" for t in tiles))

class Game2048:
    def __init__(self):
        self.board = 0
        self.score = 0
        self.reset()
        
    def reset(self):
        self.board = 0
        self.score = 0
        self.board = insert_random_tile(self.board)
        self.board = insert_random_tile(self.board)
        return self.board
        
    def step(self, action: int):
        '''
        Action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        Returns (new_board, score, done, valid_move)
        '''
        new_board = self.board
        score = 0
        if action == 0:
            new_board, score = move_up(self.board)
        elif action == 1:
            new_board, score = move_right(self.board)
        elif action == 2:
            new_board, score = move_down(self.board)
        elif action == 3:
            new_board, score = move_left(self.board)
            
        valid_move = (new_board != self.board)
        
        if valid_move:
            self.board = new_board
            self.score += score
            self.board = insert_random_tile(self.board)
            
        done = is_terminal(self.board)
        return self.board, score, done, valid_move
