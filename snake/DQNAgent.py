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
        model.add(Flatten(input_shape=self.state_space_shape[1:]))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    # def choose_action(self, state):
    #     if np.random.rand() < self.exploration_rate:
    #         return [[np.random.randint(self.action_space_size)] for _ in range(self.state_space_shape[0])]
    #     else:
    #         q_values = self.model.predict(state)
    #         return [[value] for value in np.argmax(q_values, axis=1)]

    def choose_action(self, state):
        q_values = self.model.predict(state)
        return [[value] for value in np.argmax(q_values, axis=1)]

    def update_q_values(self, state, action, reward, next_state):
        targets = reward[:, 0] + self.discount_factor * np.max(self.model.predict(next_state), axis=1)
        current_q_values = self.model.predict(state)
        for i in range(len(action)):
            current_q_values[i][action[i][0]] = targets[i]
        self.model.fit(state, current_q_values, epochs=1, verbose=0)

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
