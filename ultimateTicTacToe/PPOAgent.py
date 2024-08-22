import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm

class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=0.00005, critic_lr=0.001, clip_epsilon=0.2, gamma=0.99, lambda_=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_ = lambda_
        
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
        try:    
            probs = self.actor(state)
            # probs = self.add_noise(probs)
            probs = tf.multiply(probs, available_moves)
            
            # Normalize probabilities
            probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
            
            actions = []
            for i in range(len(probs)):
                action = np.random.choice(self.action_dim, p=np.squeeze(probs[i]))
                actions.append(action)
        except:
            print("self.actor(state): ", self.actor(state)[i])
            print("self.add_noise(probs): ", self.add_noise(self.actor(state))[i])
            print("available_moves: ", available_moves[i])
            print("tf.multiply(probs, available_moves): ", tf.multiply(self.add_noise(self.actor(state)), available_moves)[i])
            print("probs / tf.reduce_sum(probs, axis=1, keepdims=True): ", (tf.multiply(self.add_noise(self.actor(state)), available_moves) / tf.reduce_sum(tf.multiply(self.add_noise(self.actor(state)), available_moves), axis=1, keepdims=True))[i])
        
        return np.array(actions)

    def train(self, state, action, reward, next_state, done, available_actions):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)

        old_probs = self.actor(state)
        # old_probs = self.add_noise(old_probs)
        old_probs = tf.multiply(old_probs, available_actions)
        old_probs = old_probs / tf.reduce_sum(old_probs, axis=1, keepdims=True)
        old_prob = tf.gather_nd(old_probs, tf.stack([tf.range(action.shape[0]), action], axis=1))

        with tf.GradientTape() as tape2: # critic
            v = self.critic(state, training=True)
            vn = self.critic(next_state, training=True)
            vn = tf.stop_gradient(vn)
            td_target = reward + (1 - done) * self.gamma * vn
            critic_loss = tf.reduce_mean(tf.square(td_target - tf.squeeze(v)))

        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))


        v = self.critic(state, training=True)
        vn = self.critic(next_state, training=True)

        advantages = reward + self.gamma * vn * (1 - done) - v
        for _ in range(10):
            with tf.GradientTape() as tape1: # actor

                new_probs = self.actor(state, training=True)
                # new_probs = self.add_noise(new_probs)
                new_probs = tf.multiply(new_probs, available_actions)
                new_probs = new_probs / tf.reduce_sum(new_probs, axis=1, keepdims=True)
                new_prob = tf.gather_nd(new_probs, tf.stack([tf.range(action.shape[0]), action], axis=1))

                ratio = new_prob / (old_prob + 1e-10) # to avoid division by zero
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                surrogate_loss_1 = ratio * advantages
                surrogate_loss_2 = clipped_ratio * advantages
                actor_loss = -tf.reduce_mean(tf.minimum(surrogate_loss_1, surrogate_loss_2))

            grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
            
