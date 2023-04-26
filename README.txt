How to play：
1. Download and import all necessary library
2. If play minimax algorithm, run file GobangDemo(6x6 minimax).py or GobangDemo(8x8 minimax).py
3. If play MCTS with NN, run Gobang_play.py
4. If want to train AI model, run Gobang_MCTS&NN.py
5. Gobang_MCTS&NN_8x8.py and Gobang_play_8x8.py is only theoretically working, they are not fully implemented. 

How to adjust parameters:
1. Want to train model (taken 6x6 as example)
    #train_agent(games=10, mcts_simulations=3000, model_path="model.new")
    train_agent(games=1000, mcts_simulations=3000, model_path="model.new", load_model_path="model.new")
    
    Uncomment the first line under the "if __name__ == "__main__":", 
    And then adjust the second line of code, remember the number of "games" should be no less than 1000, it is the parameter to control the number of games that AI self plays. the "mcts_simulations" is how powerful you want the AI to be, the larger the stronger. Changing the path with create new model. If you want to use the same one or trainning the same one, do not change the name.

2. Want to adjust difficulty of the minimax component?
     value = minimax(alpha, beta, 4, 1)
     This should be the code on line 102 for file GobangDemo(6x6 minimax).py, the larger the harder but it will take extramely longer time for computer to calculate how to move.

3. Wand to change AI model to play game?
    In file Gobang_MCTS&NN.py, change the name of "model_path".

Instruction:
Interact with the computer in the terminal by typing command.
location (2,2) should input as: 2 2