# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:03:12 2023

@author: dongs
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Gobang:
    def __init__(self, size=6):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)

    def is_valid_move(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0

    def play_move(self, x, y, player):
        if self.is_valid_move(x, y):
            self.board[x][y] = player
            return True
        return False

    def check_winner(self, player):
        for x in range(self.size):
            for y in range(self.size):
                if (
                    self.check_horizontal(x, y, player)
                    or self.check_vertical(x, y, player)
                    or self.check_diagonal(x, y, player)
                ):
                    return True
        return False

    def check_horizontal(self, x, y, player):
        if y + 3 < self.size:
            return all(self.board[x][y + i] == player for i in range(4))
        return False

    def check_vertical(self, x, y, player):
        if x + 3 < self.size:
            return all(self.board[x + i][y] == player for i in range(4))
        return False

    def check_diagonal(self, x, y, player):
        if x + 3 < self.size and y + 3 < self.size:
            return all(self.board[x + i][y + i] == player for i in range(4))
        return False

    def is_full(self):
        return not (self.board == 0).any()

class NeuralNetwork:
    def __init__(self, input_dim, output_dim):
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=input_dim, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(output_dim, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001))

    def train(self, x, y, epochs, verbose=0):
        self.model.fit(x, y, epochs=epochs, verbose=verbose)

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, file_path):
        self.model.save_weights(file_path)

    def load_model(self, file_path):
        self.model.load_weights(file_path)

def display_board(board):
    for row in board:
        row_str = "|".join(["X" if x == 1 else "O" if x == -1 else " " for x in row])
        print(row_str)
        print("-" * (2 * len(row) - 1))

def get_ai_move(board, nn):
    possible_moves = []
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            if board[x][y] == 0:
                possible_moves.append((x, y))

    max_value = float('-inf')
    best_move = None
    for move in possible_moves:
        x, y = move
        test_board = board.copy()
        test_board[x][y] = 1
        value = nn.predict(test_board.flatten().reshape(1, -1))
        if value > max_value:
            max_value = value
            best_move = move

    return best_move

def human_vs_ai(model_path="model.txt"):
    nn = NeuralNetwork(input_dim=36, output_dim=1)
    nn.load_model(model_path)

    game = Gobang()
    player = 1

    while not game.is_full():
        if player == 1:
            x, y = get_ai_move(game.board, nn)
            game.play_move(x, y, player)
            print("AI move:")
            display_board(game.board)
            if game.check_winner(player):
                print("AI wins!")
                break
        else:
            x, y = map(int, input("Enter your move (x, y): ").split())
            if game.play_move(x, y, player):
                print("Your move:")
                display_board(game.board)
                if game.check_winner(player):
                    print("You win!")
                    break
            else:
                print("Invalid move, try again.")
                continue

        player = -player

    if not game.check_winner(1) and not game.check_winner(-1):
        print("It's a draw!")

if __name__ == "__main__":
    human_vs_ai(model_path="model.txt")
