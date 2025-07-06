import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import random

# ────────────────────────────────────────────────────────────────
# Rotation-index maps (pre-computed once at import)
# ────────────────────────────────────────────────────────────────
def _build_rotation_maps():
    """Return three numpy arrays of length 81 with the index mapping for
    90°, 180°, 270° clockwise rotations on the flattened 9×9 board.
    index = (big_r*3+big_c)*9 + (small_r*3+small_c)."""
    rot90 = np.zeros(81, dtype=int)
    rot180 = np.zeros(81, dtype=int)
    rot270 = np.zeros(81, dtype=int)
    for br in range(3):
        for bc in range(3):
            for sr in range(3):
                for sc in range(3):
                    idx = (br*3+bc)*9 + (sr*3+sc)
                    # 90° cw
                    nbr, nbc = bc, 2 - br
                    nsr, nsc = sc, 2 - sr
                    nidx90 = (nbr*3+nbc)*9 + (nsr*3+nsc)
                    # 180°
                    nidx180 = 80 - idx
                    # 270° cw
                    nidx270 = 80 - nidx90
                    rot90[idx] = nidx90
                    rot180[idx] = nidx180
                    rot270[idx] = nidx270
    return rot90, rot180, rot270

# create tf constants
ROT90, ROT180, ROT270 = [tf.constant(m, dtype=tf.int32) for m in _build_rotation_maps()]

# ────────────────────────────────────────────────────────────────
# Residual block helper
# ────────────────────────────────────────────────────────────────
def dense_res_block(x, units=256):
    y = layers.Dense(units, activation='relu')(x)
    y = layers.Dense(units)(y)
    return layers.Activation('relu')(layers.add([x, y]))

class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=0.000001, critic_lr=0.00001,
                 clip_epsilon=0.2, gamma=0.99, lambda_=0.95, entropy_weight=0.01,
                 model_path="./models/PPO", loaded=False):
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
            self.actor = keras.models.load_model(model_path + '/actor')
            self.critic = keras.models.load_model(model_path + '/critic')
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def add_noise(self, probs):
        noisy_probabilities = probs + 1e-8
        noisy_probabilities /= tf.reduce_sum(noisy_probabilities, axis=1, keepdims=True)
        return noisy_probabilities

    def act(self, state, available_moves):
        probs = self.actor.predict(state, verbose=0)
        probs = self.add_noise(probs)
        probs = tf.multiply(probs, available_moves)
        probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
        actions = tf.random.categorical(tf.math.log(probs), num_samples=1)
        return tf.squeeze(actions, axis=-1).numpy()
    
    def choose_action(self, state, available_moves, testing=False):
        return self.act(state, available_moves)

    @tf.function
    def train(self, state, action, reward, next_state, done, available_actions):
        # cast inputs
        state = tf.cast(state, tf.float32)
        next_state = tf.cast(next_state, tf.float32)
        reward = tf.cast(reward, tf.float32)
        done = tf.cast(done, tf.float32)
        action = tf.cast(action, tf.int32)
        avail = tf.cast(available_actions, tf.float32)

        # augmentation: generate rotated batches
        def rot_batch(x, mapping):
            return tf.gather(x, mapping, axis=1)

        s90 = rot_batch(state, ROT90)
        s180 = rot_batch(state, ROT180)
        s270 = rot_batch(state, ROT270)
        ns90 = rot_batch(next_state, ROT90)
        ns180 = rot_batch(next_state, ROT180)
        ns270 = rot_batch(next_state, ROT270)
        avail90 = rot_batch(avail, ROT90)
        avail180 = rot_batch(avail, ROT180)
        avail270 = rot_batch(avail, ROT270)

        a90 = tf.gather(ROT90, action)
        a180 = tf.gather(ROT180, action)
        a270 = tf.gather(ROT270, action)

        # concat along batch axis
        S = tf.concat([state, s90, s180, s270], axis=0)
        A = tf.concat([action, a90, a180, a270], axis=0)
        R = tf.concat([reward, reward, reward, reward], axis=0)
        D = tf.concat([done, done, done, done], axis=0)
        NS = tf.concat([next_state, ns90, ns180, ns270], axis=0)
        AV = tf.concat([avail, avail90, avail180, avail270], axis=0)

        # get old action probabilities
        old_probs = self.actor(S)
        old_probs = self.add_noise(old_probs)
        old_probs = tf.multiply(old_probs, AV)
        old_probs = old_probs / tf.reduce_sum(old_probs, axis=1, keepdims=True)
        old_prob = tf.gather_nd(old_probs, tf.stack([tf.range(tf.shape(A)[0]), A], axis=1))

        # Critic update
        with tf.GradientTape() as tape2:
            v = self.critic(S, training=True)
            vn = self.critic(NS, training=True)
            vn = tf.stop_gradient(vn)
            td_target = R + (1.0 - D) * self.gamma * tf.squeeze(vn)
            critic_loss = tf.reduce_mean(tf.square(td_target - tf.squeeze(v)))
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

        # advantages
        v = self.critic(S, training=True)
        vn = self.critic(NS, training=True)
        advantages = R + self.gamma * tf.squeeze(vn) * (1.0 - D) - tf.squeeze(v)

        # PPO update
        for _ in range(10):
            with tf.GradientTape() as tape1:
                new_probs = self.actor(S, training=True)
                new_probs = tf.multiply(new_probs, AV)
                new_probs = new_probs / tf.reduce_sum(new_probs, axis=1, keepdims=True)
                new_prob = tf.gather_nd(new_probs, tf.stack([tf.range(tf.shape(A)[0]), A], axis=1))

                ratio = new_prob / (old_prob + 1e-10)
                clipped = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                surrogate1 = ratio * advantages
                surrogate2 = clipped * advantages
                actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                entropy = tf.reduce_mean(tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1))
                total_loss = actor_loss + self.entropy_weight * entropy
            grads1 = tape1.gradient(total_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))

    def copy(self):
        new_agent = PPOAgent(self.state_dim, self.action_dim)
        new_agent.actor.set_weights(self.actor.get_weights())
        new_agent.critic.set_weights(self.critic.get_weights())
        return new_agent
