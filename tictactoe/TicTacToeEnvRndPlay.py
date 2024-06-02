import numpy as np

class TicTacToeEnvRndPlay:
    def __init__(self, num_games=1):
        self.num_games = num_games
        self.board_size = 3
        self.boards = np.zeros((num_games, self.board_size, self.board_size), dtype=int)
        #self.boards = np.random.choice([0, 1, -1], size=(num_games, self.board_size, self.board_size))
        self.current_player = 1  # Player 1 starts

    def reset(self, game_idx=None):
        self.boards[game_idx] = np.zeros((self.board_size, self.board_size), dtype=int)

    def step(self, moves):
        rewards_total = []
        rewards1 = []
        rewards2 = []
        dones = []
        for i in range(len(moves)): # for each game
            row = moves[i][0]
            col = moves[i][1]
            if self.boards[i, row, col] != 0:
                raise ValueError("Invalid move")
            self.boards[i, row, col] = self.current_player
            reward1, done1 = self.check_game_over(i)
            if done1:
                self.reset(i)
            rewards1.append(reward1)
        for j in range(self.boards.shape[0]):
            while True:
                row = np.random.randint(0, 3)
                col = np.random.randint(0, 3)
                if self.boards[j, row, col] == 0:
                    self.boards[j, row, col] = -self.current_player
                    break
            reward2, done2 = self.check_game_over(j)
            if done2:
                self.reset(j)
            rewards2.append(reward2)
        for k in range(len(rewards1)):
            rewards_total.append(rewards1[k] + rewards2[k])
            dones.append(done1 or done2)
        return rewards_total, dones

    def check_game_over(self, game_idx):
        board = self.boards[game_idx]
        lines = [
            board[0, :], board[1, :], board[2, :],
            board[:, 0], board[:, 1], board[:, 2],
            board.diagonal(), np.fliplr(board).diagonal()
        ]
        player = self.current_player
        for line in lines:
            if np.all(line == player):
                return 1, True  # Current player wins
            if np.all(line == -player):
                return -1, True  # Current player loses
        if np.all(board != 0):
            return 0, True  # Tie game
        return 0, False  # Game continues

    def to_state(self):
        # Returns a flattened version of the board states suitable for NN input
        states = []
        for i in range(self.num_games):
            state = np.array([])
            state = np.concatenate((state, self.boards[i].flatten()))
            states.append(state)
        return np.stack(states)

    def render(self, game_idx):
        board = self.boards[game_idx]
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        for row in board:
            print(' | '.join(symbols[x] for x in row))
            print('-' * 9)

    def get_available_moves(self):
        available_moves = np.zeros((self.boards.shape[0], self.board_size*self.board_size))
        for i in range(self.boards.shape[0]):
            board = self.boards[i].flatten()
            for j in range(len(board)):
                if board[j] == 0:
                    available_moves[i, j] = 1
        return available_moves
