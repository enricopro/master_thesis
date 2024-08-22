import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm

class A2CAgent:
    def __init__(self, state_dim, action_dim, actor_lr=0.0001, critic_lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor Network
        self.actor = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=state_dim),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='softmax')
        ])
        
        # Critic Network
        self.critic = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=state_dim),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def add_noise(self, probs):
        # # Calculate the standard normal distribution bounds
        # a, b = (0 - 0.001) / 0.000001, (np.inf - 0.001) / 0.000001

        # # Generate Truncated Gaussian noise
        # noise = truncnorm.rvs(a, b, loc=0.001, scale=0.000001, size=probs.shape)

        # # Add noise to probabilities
        # noisy_probabilities = probs + noise

        # # Ensure probabilities remain valid
        # noisy_probabilities = np.clip(noisy_probabilities, 0, 1)

        # # Normalize to ensure they sum to 1
        # noisy_probabilities /= noisy_probabilities.sum()

        noisy_probabilities = probs + 0.0000001
        noisy_probabilities /= tf.reduce_sum(noisy_probabilities, axis=1, keepdims=True)

        return noisy_probabilities

    def act(self, state, available_moves):
        probs = self.actor(state)
        probs = self.add_noise(probs)
        probs = tf.multiply(probs, available_moves)

        # Normalize probabilities
        probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)

        actions = []
        for i in range(len(probs)):
            action = np.random.choice(self.action_dim, p=np.squeeze(probs[i]))
            actions.append(action)
            
        return np.array(actions)

    def compute_loss(self, prob, td_error):
        log_prob = tf.math.log(prob + 1e-5)
        actor_loss = -log_prob * td_error
        return actor_loss

    def train(self, state, action, reward, next_state, done, available_actions):
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

            p = self.actor(state, training=True)  # Output shape: [n_games, n_actions]
            p = self.add_noise(p)
            p = tf.multiply(p, available_actions)
            p = p / tf.reduce_sum(p, axis=1, keepdims=True)
            
            prob = tf.gather_nd(p, tf.concat([tf.range(tf.shape(action)[0])[:, tf.newaxis], action[:, tf.newaxis]], axis=1))
                       
            v = self.critic(state, training=True)  # Shape: [n_games]
            vn = self.critic(next_state, training=True)  # Shape: [n_games]
            
            td_target = reward + (1 - done) * 0.99 * vn  # Shape: [n_games]
            td_error = td_target - v  # Shape: [n_games]
            
            actor_loss = self.compute_loss(prob, td_error)  # Shape: [n_games]
            actor_loss = tf.reduce_mean(actor_loss)  # Scalar

            critic_loss = tf.keras.losses.mean_squared_error(td_target, tf.squeeze(v))  # Scalar

        # Calculate gradients and update weights
        grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
        
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

