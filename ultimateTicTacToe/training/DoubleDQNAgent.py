import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf
import random

class DoubleDQNAgent:
    def __init__(self, action_space_size, state_space_shape, learning_rate=0.001, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.05, target_update_frequency=10, soft_update=False, loaded=False, model_path="./models/DDQN"):
        self.action_space_size = action_space_size
        self.state_space_shape = state_space_shape
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.target_update_frequency = target_update_frequency
        self.soft_update = soft_update
        if not loaded:
            self.model = self.build_model()
            self.target_model = self.build_model()
        else:
            print("Models loaded from memory")
            self.model = keras.models.load_model(model_path)  # Load the entire model
            self.target_model = keras.models.load_model(model_path)  # Load the entire model

        self.update_target_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_space_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        if not self.soft_update:
            self.target_model.set_weights(self.model.get_weights())
        else:
            # Get weights from both models
            target_weights = self.target_model.get_weights()
            model_weights = self.model.get_weights()

            # Perform the weighted update
            updated_weights = [0.95 * tw + 0.05 * mw for tw, mw in zip(target_weights, model_weights)]

            # Set the updated weights back to the target model
            self.target_model.set_weights(updated_weights)

    # Always choose the best action
    def act(self, state, available_moves):
        return self.choose_action(state, available_moves, True)

    def choose_action(self, state, available_actions, testing=False):
        if testing:
            q_values = self.model.predict(state, verbose=0)
            filtered_q_values = np.where(available_actions == 1, q_values, -np.inf)
            best_actions = np.argmax(filtered_q_values, axis=1)
            return best_actions
        if np.random.rand() < self.exploration_rate:
            selected_indices = np.full(available_actions.shape[0], -1, dtype=int)
            one_indices = available_actions == 1            
            
            # Get the indices of ones for each row
            indices = np.where(one_indices)

            # Split the indices array into a list of arrays, one for each row
            split_indices = np.split(indices[1], np.cumsum(np.bincount(indices[0])[:-1]))

            # Select a random index from each row's valid indices
            selected_indices = np.array([np.random.choice(idx) if len(idx) > 0 else -1 for idx in split_indices])

            return selected_indices
        else:
            q_values = self.model.predict(state, verbose=0)
            filtered_q_values = np.where(available_actions == 1, q_values, -np.inf)
            best_actions = np.argmax(filtered_q_values, axis=1)
            return best_actions

    # @tf.function
    def update_q_values(self, state, action, reward, next_state, dones, action_mask, steps):
        # Convert and cast all inputs to the correct types
        state = tf.cast(state, dtype=tf.float32)
        next_state = tf.cast(next_state, dtype=tf.float32)
        reward = tf.cast(reward, dtype=tf.float32)
        dones = tf.cast(dones, dtype=tf.float32)
        action = tf.cast(action, dtype=tf.int32)
        action_mask = tf.cast(action_mask, dtype=tf.float32)

        # Convert dones to a binary indicator (0 for done, 1 for not done)
        dones = 1.0 - dones

        self.maybe_update_target_model(steps)

        # Open a GradientTape for automatic differentiation
        with tf.GradientTape() as tape:
            # Predict Q-values for the next state using the target model
            next_state_q_values = self.target_model(next_state, training=False)

            # Filter the next state's Q-values by the action mask
            filtered_q_values = tf.where(action_mask == 1, next_state_q_values, -float('inf'))

            # Compute the maximum Q-value across the valid actions
            max_next_q_values = tf.reduce_max(filtered_q_values, axis=1)

            # Compute the target Q-values using the Bellman equation
            targets = reward + self.discount_factor * dones * max_next_q_values

            # Predict Q-values for the current state using the main model
            current_q_values = self.model(state, training=True)

            # Gather the Q-values for the actions that were taken
            action_indices = tf.stack([tf.range(tf.shape(action)[0]), action], axis=1)
            selected_q_values = tf.gather_nd(current_q_values, action_indices)

            # Compute the loss (Mean Squared Error between the predicted Q-values and target values)
            loss = tf.reduce_mean(tf.square(targets - selected_q_values))

        # Compute the gradients of the loss with respect to the model's trainable variables
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to update the model's weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if random.random() < 0.0001:
            tf.keras.backend.clear_session()


    # Call this method outside of the @tf.function
    def maybe_update_target_model(self, step):
        if step % self.target_update_frequency == 0 and not self.soft_update:
            self.update_target_model()

        if self.soft_update:
            self.update_target_model()

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def set_exploration_rate(self, new_exploration_rate):
        self.exploration_rate = new_exploration_rate
