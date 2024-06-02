import tensorflow as tf
import numpy as np

class A2CAgent:
    def __init__(self, state_dim, action_dim, actor_lr=0.001, critic_lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor Network
        self.actor = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='softmax')
        ])
        
        # Critic Network
        self.critic = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def act(self, state, available_moves):
        probs = self.actor(state)
        for i in range(probs.shape[0]):
            for j in range(probs.shape[1]):
                probs[i, j] = probs[i, j] * available_moves[i, j]
        # Normalize probabilities
        probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
        # Sample action from probability distribution
        actions = []
        for i in range(len(probs)):
            action = np.random.choice(self.action_dim, p=np.squeeze(probs[i]))
            actions.append([action])
        return np.array(actions)

    def compute_loss(self, prob, td_error):
        log_prob = tf.math.log(prob + 1e-5)
        actor_loss = -log_prob * td_error
        return actor_loss

    def train(self, state, action, reward, next_state):
        done = np.zeros(state.shape[0])
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

            p = self.actor(state, training=True)  # Output shape: [1000, 9]
            prob = tf.gather_nd(p, tf.concat([tf.range(tf.shape(action)[0])[:, tf.newaxis], action], axis=1))  # Shape: [1000]
            
            v = self.critic(state, training=True)  # Shape: [1000, 1]
            vn = self.critic(next_state, training=True)  # Shape: [1000, 1]
            
            td_target = reward + (1 - done) * 0.99 * tf.squeeze(vn)  # Shape: [1000]
            td_error = td_target - tf.squeeze(v)  # Shape: [1000]
            
            actor_loss = self.compute_loss(prob, td_error)  # Shape: [1000]
            actor_loss = tf.reduce_mean(actor_loss)  # Scalar

            critic_loss = tf.keras.losses.mean_squared_error(td_target, tf.squeeze(v))  # Scalar

        # Calculate gradients and update weights
        grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
        
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))
