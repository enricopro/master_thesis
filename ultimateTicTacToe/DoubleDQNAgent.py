import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

class DoubleDQNAgent:
    def __init__(self, action_space_size, state_space_shape, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01, update_target_every=10):
        self.action_space_size = action_space_size
        self.state_space_shape = state_space_shape
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.update_target_every = update_target_every
        self.step_counter = 0

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_space_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state, available_actions, testing=False):
        if testing or np.random.rand() > self.exploration_rate:
            q_values = self.model.predict(state, verbose=0)
            filtered_q_values = np.where(available_actions == 1, q_values, -np.inf)
            return np.argmax(filtered_q_values, axis=1)
        else:
            return np.random.choice(np.where(available_actions == 1)[0])

    def update_q_values(self, state, action, reward, next_state, done):
        target_q_values = self.target_model.predict(next_state, verbose=0)
        online_q_values_next = self.model.predict(next_state, verbose=0)
        best_actions = np.argmax(online_q_values_next, axis=1)
        
        targets = reward + self.discount_factor * (1 - done) * target_q_values[np.arange(target_q_values.shape[0]), best_actions]
        
        current_q_values = self.model.predict(state, verbose=0)
        current_q_values[np.arange(current_q_values.shape[0]), action] = targets
        
        self.model.fit(state, current_q_values, epochs=1, verbose=0)

        self.step_counter += 1
        if self.step_counter % self.update_target_every == 0:
            self.update_target_network()

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def set_exploration_rate(self, new_exploration_rate):
        self.exploration_rate = new_exploration_rate
