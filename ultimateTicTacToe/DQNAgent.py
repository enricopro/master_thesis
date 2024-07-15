import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, action_space_size, state_space_shape, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
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
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def choose_action(self, state, available_actions, testing=False):
        if testing:
            q_values = self.model.predict(state, verbose=0)
            filtered_q_values = np.where(available_actions == 1, q_values, -np.inf)
            best_actions = np.argmax(filtered_q_values, axis=1)
            return best_actions
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
            filtered_q_values = np.where(available_actions == 1, q_values, -np.inf)
            best_actions = np.argmax(filtered_q_values, axis=1)
            return best_actions
            
    def update_q_values(self, state, action, reward, next_state, dones, action_mask):
        # Convert dones to a binary indicator (0 for done, 1 for not done)
        dones = (~dones).astype(int)
        
        # Predict Q-values for the next state
        next_state_q_values = self.model.predict(next_state, verbose=0)
        # q_values = self.model.predict(state, verbose=0)
        filtered_q_values = np.where(action_mask == 1, next_state_q_values, -np.inf)
        
        # Calculate the maximum Q-value for the next state considering only valid actions
        max_next_q_values = np.max(filtered_q_values, axis=1)
        
        # Compute the target Q-value
        targets = reward + self.discount_factor * dones * max_next_q_values
        
        # Predict the current Q-values
        current_q_values = self.model.predict(state, verbose=0)
        
        # Update the Q-values with the target values only for the taken actions
        for i in range(len(action)):
            current_q_values[i][action[i]] = targets[i]
        
        # Fit the model to the updated Q-values
        self.model.fit(state, current_q_values, epochs=1, batch_size=state.shape[0], verbose=0)

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def set_exploration_rate(self, new_exploration_rate):
        self.exploration_rate = new_exploration_rate
