import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow_probability as tfp
tfd = tfp.distributions

class ReinforceAgent:
    def __init__(self, state_space_shape, learning_rate=0.001, discount_factor=0.9):
        self.state_space_shape = state_space_shape
        self.gamma = discount_factor
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_space_shape[1:]))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(9, activation='softmax'))
        model.compile(loss='mse', optimizer=self.optimizer)
        return model
    
    def choose_action(self, state):
        epsilon = 1e-8  # Small constant to prevent log(0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action_probs = self.model(state, training=False)
        mask = tf.cast(tf.equal(state, 0), dtype=tf.float32)
        action_probs = action_probs * mask
        action_probs = action_probs / tf.reduce_sum(action_probs, axis=1, keepdims=True)
        action_probs += epsilon  # Add epsilon to avoid log(0)
        log_probs = tf.math.log(action_probs)
        actions = tf.random.categorical(log_probs, num_samples=1)
        return actions.numpy()

    def learn(self, states, actions, rewards):
        epsilon = 1e-8  # Small constant to prevent log(0)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            n_steps = rewards.shape[1]
            discounts = tf.pow(self.gamma, tf.range(n_steps, dtype=tf.float32))
            returns = tf.reverse(tf.cumsum(tf.reverse(rewards * discounts[None, :], axis=[1]), axis=1), axis=[1])
            
            loss = tf.constant(0.0)
            for t, (state, action, G_t) in enumerate(zip(states, actions, returns)):
                action_probs = self.model(state, training=True)
                mask = tf.cast(tf.equal(state, 0), dtype=tf.float32)
                action_probs = action_probs * mask
                action_probs = action_probs / tf.reduce_sum(action_probs, axis=1, keepdims=True)
                action_probs += epsilon  # Add epsilon to avoid log(0)
                dist = tfd.Categorical(probs=action_probs)
                log_prob = dist.log_prob(action)
                loss -= tf.reduce_sum(log_prob * G_t)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
