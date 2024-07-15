import tensorflow as tf
import numpy as np

class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=0.0003, critic_lr=0.001, clip_epsilon=0.2, epochs=10, gamma=0.99, lambd=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.gamma = gamma
        self.lambd = lambd

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

    def act(self, state, available_moves):
        probs = self.actor(state)
        probs = tf.multiply(probs, available_moves)

        # Normalize probabilities
        probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
        # Sample action from probability distribution
        actions = []
        for i in range(len(probs)):
            action = np.random.choice(self.action_dim, p=np.squeeze(probs[i]))
            actions.append(action)
        return np.array(actions)

    def compute_advantages(self, rewards, dones, values, next_values):
        advantages = np.zeros_like(rewards)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_adv = delta + self.gamma * self.lambd * (1 - dones[t]) * last_adv
        returns = advantages + values
        return advantages, returns

    def train(self, states, actions, rewards, next_states, dones):
        values = self.critic(states)
        next_values = self.critic(next_states)
        advantages, returns = self.compute_advantages(rewards, dones, values, next_values)

        old_probs = self.actor(states)
        old_probs = tf.gather_nd(old_probs, tf.concat([tf.range(tf.shape(actions)[0])[:, tf.newaxis], actions[:, tf.newaxis]], axis=1))

        for _ in range(self.epochs):
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                probs = self.actor(states)
                probs = tf.gather_nd(probs, tf.concat([tf.range(tf.shape(actions)[0])[:, tf.newaxis], actions[:, tf.newaxis]], axis=1))
                
                ratios = probs / (old_probs + 1e-10)
                clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                
                actor_loss = -tf.reduce_mean(tf.minimum(ratios * advantages, clipped_ratios * advantages))

                values = self.critic(states)
                critic_loss = tf.keras.losses.mean_squared_error(returns, tf.squeeze(values))
                critic_loss = tf.reduce_mean(critic_loss)

            actor_grads = tape1.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            
            critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
