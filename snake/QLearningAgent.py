import numpy as np

class QLearningAgent:
    def __init__(self, action_space_size, state_space_shape, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        self.action_space_size = action_space_size
        self.state_space_shape = state_space_shape
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros(state_space_shape + (action_space_size,))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_space_size)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        current_q_value = self.q_table[state[0], action[0]]
        max_next_q_value = np.max(self.q_table[next_state])
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - current_q_value)
        self.q_table[state[0], action[0]] = new_q_value

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
