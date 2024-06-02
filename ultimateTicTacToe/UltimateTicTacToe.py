import gym
from gym import spaces
import numpy as np
import math

class UltimateTicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()  # Correct superclass initialization
        # Define action space and observation space
        self.action_space = spaces.Discrete(81)  # 81 possible actions, one for each cell in the 9x9 grid
        self.observation_space = spaces.Box(low=0, high=2, shape=(81,), dtype=np.int32)  # Board state: 0=empty, 1=player1, 2=player2

        self.board = np.zeros((9, 9), dtype=int)  # 9x9 grid
        self.global_status = np.zeros(9, dtype=int)  # Status of each 3x3 board
        self.current_player = 1  # Player 1 starts
        self.last_move = None  # Last move made

    def step(self, action):
        assert self.action_space.contains(action), "Action must be within the action space"
        global_board = math.floor(action / 27) * 3 + math.floor((action % 9) / 3)
        local_board = (math.floor(action / 9) % 3) * 3 + (action % 9 % 3)
        print("global board and local board: ", global_board, " ", local_board)
        
        if self.last_move is not None and self.last_move % 9 != global_board and self.global_status[self.last_move % 9] == 0:
            raise ValueError("Invalid move: Player must play on the designated board.")
        if self.board[global_board][local_board] != 0 or self.global_status[global_board] != 0:
            raise ValueError("Invalid move: Position is already occupied or board already won.")

        self.board[global_board][local_board] = self.current_player
        
        if self._check_win(global_board):
            self.global_status[global_board] = self.current_player
            if self._check_global_win():
                return self.board.copy(), 1, True, {}  # Player wins globally
            elif not np.any(self.global_status == 0):
                return self.board.copy(), 0, True, {}  # Global draw

        if not np.any(self.board == 0):
            return self.board.copy(), 0, True, {}  # Draw on all boards

        self.current_player = 2 if self.current_player == 1 else 1
        self._take_random_action(local_board)  # Environment takes a move

        if self._check_global_win():
            return self.board.copy(), -1, True, {}  # Environment wins globally

        if not np.any(self.board == 0):
            return self.board.copy(), 0, True, {}  # Draw

        self.current_player = 1
        return self.board.copy(), 0, False, {}

    def reset(self):
        self.board = np.zeros((9, 9), dtype=int)
        self.global_status = np.zeros(9, dtype=int)
        self.current_player = 1
        self.last_move = None
        return self.board.copy()

    def render(self, mode='human', close=False):
        board_string = ""
        for h in range(3):
            for k in range(3):
                for i in range(3):
                    if i/3 != 0:
                        board_string += "|"
                    for j in range(3):
                        board_string += str(self.board[h*3+i][k*3+j])
                board_string += "\n"
            board_string += "-+-+-+-+-+-\n"
        print(board_string)

    def _check_win(self, board_index):
        board = self.board[board_index].reshape((3, 3))
        for i in range(3):
            if np.all(board[i, :] == self.current_player) or np.all(board[:, i] == self.current_player):
                return True
        if (board[0, 0] == self.current_player and board[1, 1] == self.current_player and board[2, 2] == self.current_player) or \
           (board[0, 2] == self.current_player and board[1, 1] == self.current_player and board[2, 0] == self.current_player):
            return True
        return False

    def _check_global_win(self):
        global_board = self.global_status.reshape((3, 3))
        # Check rows, columns, and diagonals for a win in the global board
        for i in range(3):
            if np.all(global_board[i, :] == self.current_player) or np.all(global_board[:, i] == self.current_player):
                return True
        if (global_board[0, 0] == self.current_player and global_board[1, 1] == self.current_player and global_board[2, 2] == self.current_player) or \
           (global_board[0, 2] == self.current_player and global_board[1, 1] == self.current_player and global_board[2, 0] == self.current_player):
            return True
        return False

    def _take_random_action(self, previous_board):
        next_board = previous_board
        if np.all(self.board[previous_board]) != 0: # if the local board is full
            while np.all(self.board[next_board]) != 0:
                next_board = np.random.randint(0, 9)
        next_local_board = np.random.randint(0, 9)
        while self.board[next_board][next_local_board] != 0: #TODO can be improved by sampling the possible moves
            next_local_board = np.random.randint(0, 9)
        # Perform the move
        self.board[next_board][next_local_board] = self.current_player
        self.current_player = -self.current_player
        # Setting the last move as a number between 0 and 80 included
        self.last_move = 27 * (next_board // 3) + 9 * (next_local_board // 3) + 3 * (next_board % 3) + (next_local_board % 3)

    def to_state(self):
        flat_board = []
        # Flatten the board into a single vector
        for h in range(3):
            for k in range(3):
                for i in range(3):
                    for j in range(3):
                        flat_board.append(self.board[h*3+i][k*3+j])
        flat_board = np.array(flat_board)
        available_moves = np.zeros(81, dtype=int)  # Initialize all to unavailable
        if self.last_move == None:
            return flat_board, available_moves+1
        start_index = math.floor(self.last_move / 9) * 9
        end_index = start_index + 8
        print("start index: ", start_index, " end index: ", end_index)
        if not np.all(flat_board[start_index:end_index]) != 0:
            for i in range(start_index, end_index+1):
                if flat_board[i] == 0:
                    available_moves[i] = 1
        else: 
            for i in range(len(flat_board)):
                if flat_board[i] == 0:
                    available_moves[i] = 1
        
        return flat_board, available_moves