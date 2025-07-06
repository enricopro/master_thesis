import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm
from tensorflow import keras
import random
from tensorflow.keras import layers, models, optimizers

def dense_res_block(x, units=256):
    y = layers.Dense(units, activation='relu')(x)
    y = layers.Dense(units)(y)
    return layers.Activation('relu')(layers.add([x, y]))

class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=0.000001, critic_lr=0.00001, clip_epsilon=0.2, gamma=0.99, lambda_=0.95, entropy_weight=0.01, model_path="./models/PPO", loaded=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_weight = entropy_weight
        
        if not loaded:
            print("Models created from scratch")
            # ── build actor net ──
            a_in = layers.Input(shape=self.state_dim)
            ax = layers.Dense(256, activation='relu')(a_in)
            for _ in range(6):
                ax = dense_res_block(ax, 256)
            ax = layers.LayerNormalization()(ax)
            a_out = layers.Activation('softmax')(layers.Dense(action_dim)(ax))
            self.actor = models.Model(a_in, a_out, name="actor")

            # ── build critic net ──
            c_in = layers.Input(shape=self.state_dim)
            cx = layers.Dense(256, activation='relu')(c_in)
            for _ in range(6):
                cx = dense_res_block(cx, 256)
            cx = layers.LayerNormalization()(cx)
            c_out = layers.Dense(1)(cx)
            self.critic = models.Model(c_in, c_out, name="critic")

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
        try:    
            probs = self.actor.predict(state, verbose=0)
            probs = self.add_noise(probs)
            probs = tf.multiply(probs, available_moves)
            
            # Normalize probabilities
            probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
            
            # Use TensorFlow's tf.random.categorical to select actions based on the probabilities
            actions = tf.random.categorical(tf.math.log(probs), num_samples=1)
            actions = tf.squeeze(actions, axis=-1)  # Remove unnecessary dimensions

        except:
            print("self.actor(state): ", self.actor(state)[i])
            print("self.add_noise(probs): ", self.add_noise(self.actor(state))[i])
            print("available_moves: ", available_moves[i])
            print("tf.multiply(probs, available_moves): ", tf.multiply(self.add_noise(self.actor(state)), available_moves)[i])
            print("probs / tf.reduce_sum(probs, axis=1, keepdims=True): ", (tf.multiply(self.add_noise(self.actor(state)), available_moves) / tf.reduce_sum(tf.multiply(self.add_noise(self.actor(state)), available_moves), axis=1, keepdims=True))[i])
        
        return actions.numpy()
    
    def choose_action(self, state, available_moves, testing=False):
        return self.act(state, available_moves)

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

        # Get old probabilities
        old_probs = self.actor(state)
        old_probs = self.add_noise(old_probs)
        old_probs = tf.multiply(old_probs, available_actions)
        old_probs = old_probs / tf.reduce_sum(old_probs, axis=1, keepdims=True)
        old_prob = tf.gather_nd(old_probs, tf.stack([tf.range(action.shape[0]), action], axis=1))

        # Critic loss calculation
        with tf.GradientTape() as tape2:
            v = self.critic(state, training=True)
            vn = self.critic(next_state, training=True)
            vn = tf.stop_gradient(vn)
            td_target = reward + (1 - done) * self.gamma * vn
            critic_loss = tf.reduce_mean(tf.square(td_target - tf.squeeze(v)))

        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

        # Calculate advantages
        v = self.critic(state, training=True)
        vn = self.critic(next_state, training=True)
        advantages = reward + self.gamma * vn * (1 - done) - v

        # PPO training loop
        for _ in range(10):
            with tf.GradientTape() as tape1:
                new_probs = self.actor(state, training=True)
                new_probs = tf.multiply(new_probs, available_actions)
                new_probs = new_probs / tf.reduce_sum(new_probs, axis=1, keepdims=True)
                new_prob = tf.gather_nd(new_probs, tf.stack([tf.range(action.shape[0]), action], axis=1))

                # Compute the PPO ratio
                ratio = new_prob / (old_prob + 1e-10)
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                surrogate_loss_1 = ratio * advantages
                surrogate_loss_2 = clipped_ratio * advantages
                actor_loss = -tf.reduce_mean(tf.minimum(surrogate_loss_1, surrogate_loss_2))

                # Compute entropy to encourage exploration
                entropy = tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1)
                entropy_loss = tf.reduce_mean(entropy)

                # Add entropy loss to the actor loss
                total_actor_loss = actor_loss + self.entropy_weight * entropy_loss

            # Apply gradients for the actor network
            grads1 = tape1.gradient(total_actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
            
        # Clear the tape to free memory
        del tape1, tape2, grads1, grads2
                
    def copy(self):
        new_agent = PPOAgent(self.state_dim, self.action_dim)
        new_agent.actor.set_weights(self.actor.get_weights())
        new_agent.critic.set_weights(self.critic.get_weights())
        return new_agent