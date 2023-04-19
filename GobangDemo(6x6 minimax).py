#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:16:46 2023

@author: thomaslyu
"""

import numpy as np

# Define the size of the game board and the win condition
board_size = 6
win_condition = 4

# Define the possible values for each position on the board
# 0: empty, 1: player 1, -1: player 2 (AI)
player_symbols = {0: '-', 1: 'X', -1: 'O'}

# Initialize the game board
board = np.zeros((board_size, board_size), dtype=int)

# Define a function to check if a player has won the game
def check_win(player):
    # Check rows
    for i in range(board_size):
        for j in range(board_size - win_condition + 1):
            if all(board[i, j+jj] == player for jj in range(win_condition)):
                return True
    # Check columns
    for i in range(board_size - win_condition + 1):
        for j in range(board_size):
            if all(board[i+ii, j] == player for ii in range(win_condition)):
                return True
    # Check diagonal (top-left to bottom-right)
    for i in range(board_size - win_condition + 1):
        for j in range(board_size - win_condition + 1):
            if all(board[i+ii, j+jj] == player for ii, jj in zip(range(win_condition), range(win_condition))):
                return True
    # Check diagonal (bottom-left to top-right)
    for i in range(win_condition - 1, board_size):
        for j in range(board_size - win_condition + 1):
            if all(board[i-ii, j+jj] == player for ii, jj in zip(range(win_condition), range(win_condition))):
                return True
    return False

# Define a function to print the current game board
def print_board():
    for i in range(board_size):
        print(' '.join([player_symbols[board[i, j]] for j in range(board_size)]))

# Define a function to get the current board state as a string
def get_board_state():
    return ''.join([player_symbols[board[i, j]] for i in range(board_size) for j in range(board_size)])

# Define a function to get the possible moves for the AI player
def get_possible_moves():
    return [(i, j) for i in range(board_size) for j in range(board_size) if board[i, j] == 0]

# Define a function to evaluate the current board state for the AI player
def evaluate_board():
    if check_win(-1):
        return 100
    elif check_win(1):
        return -100
    else:
        return 0

# Define the minimax algorithm to select the best move for the AI player
# Define the minimax algorithm with alpha-beta pruning to select the best move for the AI player
def minimax(alpha, beta, depth, player):
    if depth == 0 or check_win(1) or check_win(-1):
        return evaluate_board()
    if player == -1:
        best_value = -np.Inf
        for move in get_possible_moves():
            board[move] = player
            value = minimax(alpha, beta, depth-1, 1)
            board[move] = 0
            best_value = max(best_value, value)
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
        return best_value
    else:
        best_value = np.Inf
        for move in get_possible_moves():
            board[move] = player
            value = minimax(alpha, beta, depth-1, -1)
            board[move] = 0
            best_value = min(best_value, value)
            beta = min(beta, best_value)
            if alpha >= beta:
                break
        return best_value


def get_ai_move():
    alpha = -np.Inf
    beta = np.Inf
    best_move = None
    for move in get_possible_moves():
        board[move] = -1
        value = minimax(alpha, beta, 4, 1) # Set the search depth here (higher depth = stronger AI)
        board[move] = 0
        if value > alpha:
            alpha = value
            best_move = move
    return best_move

# Play the game
print("Welcome to Gobang!")
print("You are player 1 (X). The AI is player 2 (O).")
print_board()

while True:
    # Player 1's turn (human player)
    print("Player 1's turn.")
    row, col = map(int, input("Enter row and column (0-5): ").split())
    while board[row, col] != 0:
        print("That position is already occupied. Try again.")
        row, col = map(int, input("Enter row and column (0-5): ").split())
    board[row, col] = 1
    print_board()
    if check_win(1):
        print("Player 1 wins!")
        break
    elif len(get_possible_moves()) == 0:
        print("It's a tie!")
        break

    # Player 2's turn (AI player)
    print("Player 2's turn.")
    row, col = get_ai_move()
    board[row, col] = -1
    print_board()
    if check_win(-1):
        print("Player 2 wins!")
        break
    elif len(get_possible_moves()) == 0:
        print("It's a tie!")
        break
