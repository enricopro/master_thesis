import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Flatten

class DQNAgent:
    def __init__(self, action_space_size, state_space_shape, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        self.action_space_size = action_space_size
        self.state_space_shape = state_space_shape
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_space_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def choose_action(self, state, available_actions):
        if np.random.rand() < self.exploration_rate:
            selected_indices = np.full(available_actions.shape[0], -1, dtype=int)
            one_indices = available_actions == 1
            # For each row where there is at least one '1', select a random index of '1'
            for i in range(available_actions.shape[0]):
                valid_indices = np.where(one_indices[i])[0]
                if valid_indices.size > 0:
                    selected_indices[i] = np.random.choice(valid_indices)
            return selected_indices
        else:
            q_values = self.model.predict(state, verbose=0)
            return [np.argmax(q_values[i]+available_actions[i]*100000) for i in range(len(available_actions))]
            
    def update_q_values(self, state, action, reward, next_state):
        targets = reward + self.discount_factor * np.max(self.model.predict(next_state, verbose=0), axis=1)
        current_q_values = self.model.predict(state, verbose=0)
        for i in range(len(action)):
            current_q_values[i][action[i]] = targets[i]
        self.model.fit(state, current_q_values, epochs=1, verbose=0)

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
