import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm
from tensorflow import keras
import random

class A2CAgent:
    def __init__(self, state_dim, action_dim, actor_lr=0.0003, critic_lr=0.0001, entropy_weight=0.01, model_path="./models/A2C", loaded=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.entropy_weight = entropy_weight
        if not loaded:
            print("Models created from scratch")
            # Actor Network
            self.actor = tf.keras.models.Sequential([
                tf.keras.layers.Dense(256, use_bias=False, input_shape=state_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dense(256, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dense(action_dim, activation='softmax')  # Output layer with softmax
            ])

            # Critic Network
            self.critic = tf.keras.models.Sequential([
                tf.keras.layers.Dense(256, use_bias=False, input_shape=state_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dense(256, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dense(1)  # Output layer without activation
            ])

                    
        else:
            print("Models loaded from memory")
            self.actor = keras.models.load_model(model_path + '/actor')  # Load the entire model
            self.critic = keras.models.load_model(model_path + '/critic')  # Load the entire model
            
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def add_noise(self, probs):
        noisy_probabilities = probs + 0.00000001
        noisy_probabilities /= tf.reduce_sum(noisy_probabilities, axis=1, keepdims=True)

        return noisy_probabilities

    def act(self, state, available_moves):
        # Forward pass through the actor network
        probs = self.actor.predict(state, verbose=0)
        probs = self.add_noise(probs)
        probs = tf.multiply(probs, available_moves)

        # Normalize probabilities
        probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
        
        # Use TensorFlow's tf.random.categorical to select actions based on the probabilities
        actions = tf.random.categorical(tf.math.log(probs), num_samples=1)
        actions = tf.squeeze(actions, axis=-1)  # Remove unnecessary dimensions

        # Clean up unnecessary tensors to free memory
        del probs

        return actions.numpy()

    def compute_loss(self, prob, td_error, p):
        # Compute the standard actor loss
        log_prob = tf.math.log(prob + 1e-5)
        actor_loss = -log_prob * td_error
        
        # Compute entropy (to encourage exploration)
        entropy = -tf.reduce_sum(p * tf.math.log(p + 1e-5), axis=1)
        
        # Combine actor loss with entropy loss
        total_loss = tf.reduce_mean(actor_loss) - self.entropy_weight * tf.reduce_mean(entropy)
        
        return total_loss, entropy

    @tf.function
    def train(self, state, action, reward, next_state, done, available_actions):
        
        state = tf.cast(state, dtype=tf.float32)
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        next_state = tf.cast(next_state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)

        reward = tf.cast(reward, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        done = tf.cast(done, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)

        action = tf.cast(action, dtype=tf.int32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        
        available_actions = tf.cast(available_actions, dtype=tf.float32)
        available_actions = tf.convert_to_tensor(available_actions, dtype=tf.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # Actor forward pass
            p = self.actor(state, training=True)  # Output shape: [n_games, n_actions]
            p = self.add_noise(p)
            p = tf.multiply(p, available_actions)
            p = p / tf.reduce_sum(p, axis=1, keepdims=True)  # Normalize probabilities

            # Gather the probabilities of the taken actions
            prob = tf.gather_nd(p, tf.concat([tf.range(tf.shape(action)[0])[:, tf.newaxis], action[:, tf.newaxis]], axis=1))

            # Critic forward pass
            v = self.critic(state, training=True)  # Critic's value estimate for current state
            vn = self.critic(next_state, training=True)  # Critic's value estimate for next state

            # TD target and TD error calculations
            td_target = reward + (1 - done) * 0.99 * vn  # Discount factor 0.99
            td_error = td_target - v  # Temporal Difference error

            # Compute the combined actor loss with entropy
            actor_loss, entropy = self.compute_loss(prob, td_error, p)
            
            # Critic loss (mean squared error)
            critic_loss = tf.keras.losses.mean_squared_error(td_target, tf.squeeze(v))  # Scalar

        # Calculate actor gradients
        grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
        
        # Calculate critic gradients
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

        # Clear the session to free memory
        if random.random() < 0.00001:
            tf.keras.backend.clear_session()

        del tape1, tape2, grads1, grads2

    def copy(self):
        new_agent = A2CAgent(self.state_dim, self.action_dim)
        new_agent.actor.set_weights(self.actor.get_weights())
        new_agent.critic.set_weights(self.critic.get_weights())
        return new_agent