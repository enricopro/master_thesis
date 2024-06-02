import numpy as np

class TicTacToeEnv:
    def __init__(self, num_games=1):
        self.num_games = num_games
        self.board_size = 3
        self.boards = np.zeros((num_games, self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # Player 1 starts

    def reset(self, game_idx=None):
        self.boards[game_idx] = np.zeros((self.board_size, self.board_size), dtype=int)

    def step(self, moves):
        rewards = []
        dones = []
        for i in range(len(moves)):
            row = moves[i][0]
            col = moves[i][1]
            if self.boards[i, row, col] != 0:
                reward = -5 # Invalid move
                done = False
            else:
                self.boards[i, row, col] = self.current_player
                reward, done = self.check_game_over(i)
                if not done:
                    self.current_player = - self.current_player  # Switch player
                else:
                    self.reset(i)
            rewards.append(reward)
            dones.append(done)
        return rewards

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
                return 10, True  # Current player wins
        if np.all(board != 0):
            return -0.5, True  # Tie game
        return 0, False  # Game continues

    def to_state(self):
        # Returns a flattened version of the board states suitable for NN input
        states = []
        for i in range(self.num_games):
            state = np.array([self.current_player])
            state = np.concatenate((state, self.boards[i].flatten()))
            states.append(state)
        return np.stack(states)

    def render(self, game_idx):
        board = self.boards[game_idx]
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        for row in board:
            print(' | '.join(symbols[x] for x in row))
            print('-' * 9)
