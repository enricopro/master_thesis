import gym
from gym import spaces
import numpy as np
import math

class UltimateTicTacToeEnvSelfPlay(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()  # Ensure proper superclass initialization
        self.action_space = spaces.Discrete(81)  # 81 possible actions, one for each cell in the 9x9 grid
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9, 9), dtype=int)  # Define observation space
        
        # Initialize state variables
        self.board = np.zeros((9, 9), dtype=int)  # 9x9 grid
        self.sub_boards_won = np.zeros(9, dtype=int)  # Track wins on each sub-board
        self.current_player = 1  # Player 1 starts
        self.last_move = None  # Store the last move

    def step(self, action):
        reward = 0
        # Check if the action is valid
        if action < 0 or action > 80:
            raise ValueError(f"Invalid action {action}")
        # Check if the action is valid
        valid_actions = self.get_valid_actions()
        if valid_actions[action] == 0:
            raise ValueError(f"Invalid action {action}")
        # Update the board
        board_index = math.floor(action / 9)
        cell_index = action % 9
        self.board[board_index, cell_index] = self.current_player
        self.last_move = action
        # Check if the player has won
        done = self.check_win(board_index)
        global_done = False
        if done:
            reward += 0 # we can adjust the reward as needed
            self.sub_boards_won[board_index] = self.current_player
            # Check if the global board is won
            global_done = self.check_global_win()
            if global_done:
                reward += 1
                self.reset()
                return self.board.copy(), reward, global_done, {}
        if np.all(self.get_valid_actions() == 0):
            reward += 0
            self.reset()
            return self.board.copy(), reward, True, {}

        # Switch player
        self.current_player = -self.current_player

        return self.board.copy(), reward, global_done, {}

    def reset(self):
        self.board = np.zeros((9, 9), dtype=int)
        self.current_player = -self.current_player # TODO I'm not sure about it
        self.sub_boards_won = np.zeros(9, dtype=int)
        self.last_move = None

    def render(self, mode='human', close=False):
        board_string = ""
        for h in range(3):
            for k in range(3):
                for i in range(3):
                    if i/3 != 0:
                        board_string += "|"
                    for j in range(3):
                        if self.board[h*3+i][k*3+j] == 0:
                            board_string += " "
                        if self.board[h*3+i][k*3+j] == 1:
                            board_string += "X"
                        if self.board[h*3+i][k*3+j] == -1:
                            board_string += "O"
                board_string += "\n"
            board_string += "-+-+-+-+-+-\n"
        print(board_string)

    def check_win(self, board_index):
        # Check if the player has won the board
        board = self.board[board_index].reshape((3, 3))
        for i in range(3):
            if np.all(board[i, :] == self.current_player) or np.all(board[:, i] == self.current_player):
                return True
        if (board[0, 0] == self.current_player and board[1, 1] == self.current_player and board[2, 2] == self.current_player) or \
              (board[0, 2] == self.current_player and board[1, 1] == self.current_player and board[2, 0] == self.current_player):
                return True
        return False

    def check_global_win(self):
        # Check if the player has won the global board
        global_board = self.sub_boards_won.reshape((3, 3))
        for i in range(3):
            if np.all(global_board[i, :] == self.current_player) or np.all(global_board[:, i] == self.current_player):
                return True
        if (global_board[0, 0] == self.current_player and global_board[1, 1] == self.current_player and global_board[2, 2] == self.current_player) or \
              (global_board[0, 2] == self.current_player and global_board[1, 1] == self.current_player and global_board[2, 0] == self.current_player):
                return True
        return False

    def take_random_action(self):
        valid_actions = self.get_valid_actions()
        action = np.random.choice(np.where(valid_actions == 1)[0])
        board_index = math.floor(action / 9)
        cell_index = action % 9
        self.board[board_index, cell_index] = self.current_player
        self.last_move = action

    def get_valid_actions(self):
        flatten_board = self.board.flatten()
        valid_actions = np.zeros(81, dtype=int)
        # If last action is None, all actions are valid
        if self.last_move is None:
            valid_actions += 1
            return valid_actions
        # Check if the corresponding board has been won
        previous_cell = self.last_move % 9
        starting_index = previous_cell * 9
        ending_index = starting_index + 9
        if self.sub_boards_won[previous_cell] != 0: # the board has been won
            for i in range(len(flatten_board)):
                if self.sub_boards_won[math.floor(i / 9)] != 0:
                    continue
                if flatten_board[i] == 0:
                    valid_actions[i] = 1
            return valid_actions
        if (flatten_board[starting_index:ending_index] != 0).all(): # the board is full
            for i in range(len(flatten_board)):
                if self.sub_boards_won[math.floor(i / 9)] != 0:
                    continue
                if flatten_board[i] == 0:
                    valid_actions[i] = 1
            return valid_actions
        # Case where the corresponding board is not full
        for i in range(starting_index, ending_index):
            if flatten_board[i] == 0:
                valid_actions[i] = 1
        
        return valid_actions

    def to_state(self):
        return self.board.flatten()*self.current_player, self.get_valid_actions()