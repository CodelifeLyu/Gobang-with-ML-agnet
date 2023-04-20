import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import pickle
import math
import time


class Gobang:
    def __init__(self, size=8):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)

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
        if y + 4 < self.size:
            return all(self.board[x][y + i] == player for i in range(5))
        return False

    def check_vertical(self, x, y, player):
        if x + 4 < self.size:
            return all(self.board[x + i][y] == player for i in range(5))
        return False

    def check_diagonal(self, x, y, player):
        if x + 4 < self.size and y + 4 < self.size:
            return all(self.board[x + i][y + i] == player for i in range(5))
        return False

    def is_full(self):
        return not (self.board == 0).any()


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = self.get_valid_actions()

    def get_valid_actions(self):
        actions = []
        for x in range(self.state.size):
            for y in range(self.state.size):
                if self.state.is_valid_move(x, y):
                    actions.append(x * self.state.size + y)
        return actions

    def uct_select_child(self):
        s = sorted(self.children, key=lambda c: c.wins / c.visits + math.sqrt(2 * math.log(self.visits) / c.visits))[-1]
        return s

    def add_child(self, action, state):
        child = Node(state, parent=self)
        self.untried_actions.remove(action)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result


class MCTS:
    def __init__(self, exploration_param=0.8):
        self.exploration_param = exploration_param

    def search(self, root_state, max_iter, nn, temperature=1):
        root_node = Node(root_state)

        for _ in range(max_iter):
            node = root_node
            state = Gobang(size=root_state.size)
            state.board = root_state.board.copy()

            # Selection
            while node.untried_actions == [] and node.children != []:
                node = node.uct_select_child()
                last_action = node.parent.children.index(node)
                x, y = last_action // state.size, last_action % state.size
                player = 1 if (state.board != 0).sum() % 2 == 0 else -1
                state.play_move(x, y, player)

            # Expansion
            if node.untried_actions != []:
                action = random.choice(node.untried_actions)
                x, y = action // state.size, action % state.size
                player = 1 if (state.board != 0).sum() % 2 == 0 else -1
                state.play_move(x, y, player)
                node = node.add_child(action, state)

            # Simulation
            while not state.is_full():
                valid_actions = []
                for x in range(state.size):
                    for y in range(state.size):
                        if state.is_valid_move(x, y):
                            valid_actions.append((x, y))

                if valid_actions:
                    x, y = random.choice(valid_actions)
                    player = 1 if (state.board != 0).sum() % 2 == 0 else -1
                    state.play_move(x, y, player)
                else:
                    break

            # Backpropagation
            while node is not None:
                node.update(state.board.sum())
                node = node.parent

        # Choose the best move based on visit count
        most_visited_child = sorted(root_node.children, key=lambda c: c.visits)[-1]
        best_action = root_node.children.index(most_visited_child)

        return best_action


class NeuralNetwork:
    def __init__(self, input_dim, output_dim):
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=input_dim, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(output_dim, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.01))

    def train(self, x, y, epochs, verbose=0):
        self.model.fit(x, y, epochs=epochs, verbose=verbose)

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, file_path):
        self.model.save_weights(file_path)

    def load_model(self, file_path):
        self.model.load_weights(file_path)


def train_agent(games=1000, mcts_simulations=100, model_path="model8x8", load_model_path=None):
    gobang = Gobang(size=8)
    memory = []

    # Initialize neural network
    nn = NeuralNetwork(input_dim=gobang.size * gobang.size, output_dim=1)
    
    # Load model from file if provided
    if load_model_path is not None:
        nn.load_model(load_model_path)
        
    #start timer
    start_time = time.time()

    # Train the agent for the specified number of games
    for game in range(games):
        gobang.reset()
        mcts = MCTS()

        # Play the game until it ends
        while not gobang.is_full():
            # Perform MCTS simulations to find the best move
            board_state = gobang.board.copy()
            action = mcts.search(gobang, mcts_simulations, nn)
            x, y = action // gobang.size, action % gobang.size
            player = 1 if (gobang.board != 0).sum() % 2 == 0 else -1

            # Play the move and check for a winner
            gobang.play_move(x, y, player)
            if gobang.check_winner(player):
                reward = 1 if player == 1 else -1
                break
            else:
                reward = 0

            # Store the board state, action, and reward in memory
            memory.append((board_state, action, reward))

        # Train neural network using experience replay
        if len(memory) >= 1000:
            minibatch = random.sample(memory, 1000)

            x_train = []
            y_train = []
            for board_state, action, reward in minibatch:
                x_train.append(board_state.flatten())
                y_train.append(reward)

            x_train = np.array(x_train)
            y_train = np.array(y_train).reshape(-1, 1)

            nn.train(x_train, y_train, epochs=10)

        # Print progress
        if (game + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Game {game + 1}/{games} completed. Time elapsed: {elapsed_time:.2f} seconds.")

    # Save the trained model to file
    nn.save_model(model_path)
    print(f"Trained model saved to {model_path}")
    
    total_elapsed_time = time.time() - start_time
    print(f"Total training time: {total_elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    #initial model
    #train_agent(games=100, mcts_simulations=100, model_path="model8x8")
    train_agent(games=1000, mcts_simulations=1000, model_path="model8x8", load_model_path="model8x8")

