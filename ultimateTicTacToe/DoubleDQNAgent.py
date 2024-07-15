import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

class DoubleDQNAgent:
    def __init__(self, action_space_size, state_space_shape, learning_rate=0.001, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01, target_update_frequency=10):
        self.action_space_size = action_space_size
        self.state_space_shape = state_space_shape
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.target_update_frequency = target_update_frequency
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_space_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state, available_actions, testing=False):
        if testing:
            q_values = self.model.predict(state, verbose=0)
            filtered_q_values = np.where(available_actions == 1, q_values, -np.inf)
            best_actions = np.argmax(filtered_q_values, axis=1)
            return best_actions
        if np.random.rand() < self.exploration_rate:
            selected_indices = np.full(available_actions.shape[0], -1, dtype=int)
            one_indices = available_actions == 1
            for i in range(available_actions.shape[0]):
                valid_indices = np.where(one_indices[i])[0]
                if valid_indices.size > 0:
                    selected_indices[i] = np.random.choice(valid_indices)
            return selected_indices
        else:
            q_values = self.model.predict(state, verbose=0)
            filtered_q_values = np.where(available_actions == 1, q_values, -np.inf)
            best_actions = np.argmax(filtered_q_values, axis=1)
            return best_actions

    def update_q_values(self, state, action, reward, next_state, dones, action_mask, step):
        dones = (~dones).astype(int)
        next_state_q_values = self.target_model.predict(next_state, verbose=0)
        filtered_q_values = np.where(action_mask == 1, next_state_q_values, -np.inf)
        max_next_q_values = np.max(filtered_q_values, axis=1)
        targets = reward + self.discount_factor * dones * max_next_q_values
        current_q_values = self.model.predict(state, verbose=0)
        for i in range(len(action)):
            current_q_values[i][action[i]] = targets[i]
        self.model.fit(state, current_q_values, epochs=1, batch_size=state.shape[0], verbose=0)
        
        if step % self.target_update_frequency == 0:
            self.update_target_model()

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def set_exploration_rate(self, new_exploration_rate):
        self.exploration_rate = new_exploration_rate
