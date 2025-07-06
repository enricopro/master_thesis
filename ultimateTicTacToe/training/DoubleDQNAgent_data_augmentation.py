import numpy as np
import tensorflow as tf
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
                    nidx180 = 80 - idx  # quick because 3×3×9 symmetric
                    # 270° cw == 90° ccw
                    nidx270 = 80 - nidx90
                    rot90[idx] = nidx90
                    rot180[idx] = nidx180
                    rot270[idx] = nidx270
    return rot90, rot180, rot270

ROT90, ROT180, ROT270 = [tf.constant(m, dtype=tf.int32) for m in _build_rotation_maps()]

# ────────────────────────────────────────────────────────────────
# Residual block helper
# ────────────────────────────────────────────────────────────────

def dense_residual_block(x, units=256):
    y = layers.Dense(units, activation='relu')(x)
    y = layers.Dense(units)(y)
    return layers.Activation('relu')(layers.add([x, y]))

# ────────────────────────────────────────────────────────────────
# Double-DQN agent with rotation augmentation
# ────────────────────────────────────────────────────────────────

class DoubleDQNAgent:
    def __init__(self, action_space_size, state_space_shape, learning_rate=3e-4,
                 discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99,
                 min_exploration_rate=0.05, target_update_frequency=10,
                 soft_update=False, loaded=False, model_path="./models/DDQN"):

        self.action_space_size = action_space_size
        self.state_space_shape = state_space_shape
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.target_update_frequency = target_update_frequency
        self.soft_update = soft_update

        if loaded:
            print("Models loaded from", model_path)
            self.model = tf.keras.models.load_model(model_path)
            self.target_model = tf.keras.models.load_model(model_path)
        else:
            self.model = self._build_model()
            self.target_model = self._build_model()
        self.update_target_model()
        self.optimizer = optimizers.Adam(self.learning_rate)

    # ────────────────────────────────────────────────────────────
    # Network
    # ────────────────────────────────────────────────────────────
    def _build_model(self):
        inp = layers.Input(shape=self.state_space_shape)
        x = layers.Dense(256, activation='relu')(inp)
        for _ in range(6):
            x = dense_residual_block(x, 256)
        x = layers.LayerNormalization()(x)
        out = layers.Dense(self.action_space_size, activation='linear')(x)
        model = models.Model(inp, out)
        model.compile(optimizer=optimizers.Adam(self.learning_rate), loss='mse')
        return model

    # ────────────────────────────────────────────────────────────
    # Target-net utilities
    # ────────────────────────────────────────────────────────────
    def update_target_model(self):
        if self.soft_update:
            tw = self.target_model.get_weights()
            mw = self.model.get_weights()
            self.target_model.set_weights([0.95*t + 0.05*m for t, m in zip(tw, mw)])
        else:
            self.target_model.set_weights(self.model.get_weights())

    def maybe_update_target_model(self, step):
        if not self.soft_update and step % self.target_update_frequency == 0:
            self.update_target_model()
        if self.soft_update:
            self.update_target_model()

    # ────────────────────────────────────────────────────────────
    # ε-greedy action selection
    # ────────────────────────────────────────────────────────────
    def choose_action(self, state, avail, test=False):
        if test or random.random() > self.exploration_rate:
            q = self.model.predict(state, verbose=0)
            q = np.where(avail == 1, q, -np.inf)
            return np.argmax(q, axis=1)
        # random valid move per batch element
        sel = np.full(avail.shape[0], -1, int)
        idxs = np.where(avail == 1)
        splits = np.split(idxs[1], np.cumsum(np.bincount(idxs[0])[:-1]))
        sel[:] = [np.random.choice(s) if len(s) else -1 for s in splits]
        return sel

    act = lambda self, s, a: self.choose_action(s, a, True)

    # ────────────────────────────────────────────────────────────
    # Core update with rotation augmentation
    # ────────────────────────────────────────────────────────────
    def update_q_values(self, s, a, r, s2, done, mask2, step):
        # tf tensors
        s   = tf.cast(s,   tf.float32)
        s2  = tf.cast(s2,  tf.float32)
        r   = tf.cast(r,   tf.float32)
        d   = 1.0 - tf.cast(done, tf.float32)
        a   = tf.cast(a,   tf.int32)
        m2  = tf.cast(mask2, tf.float32)

        # augment: original + 90 + 180 + 270
        def rot_batch(batch, mapping):
            return tf.gather(batch, mapping, axis=1)

        s90, s180, s270   = rot_batch(s, ROT90), rot_batch(s, ROT180), rot_batch(s, ROT270)
        s2_90, s2_180, s2_270 = rot_batch(s2, ROT90), rot_batch(s2, ROT180), rot_batch(s2, ROT270)
        m2_90, m2_180, m2_270 = rot_batch(m2, ROT90), rot_batch(m2, ROT180), rot_batch(m2, ROT270)

        a90   = tf.gather(ROT90,  a)
        a180  = tf.gather(ROT180, a)
        a270  = tf.gather(ROT270, a)

        # concat along batch axis
        S      = tf.concat([s,  s90,  s180,  s270], 0)
        A      = tf.concat([a,  a90,  a180,  a270], 0)
        R      = tf.concat([r,  r,    r,    r],    0)
        D      = tf.concat([d,  d,    d,    d],    0)
        S2     = tf.concat([s2, s2_90, s2_180, s2_270], 0)
        M2     = tf.concat([m2, m2_90, m2_180, m2_270], 0)

        self.maybe_update_target_model(step)

        with tf.GradientTape() as tape:
            q_next = self.target_model(S2)
            q_next = tf.where(M2 == 1, q_next, -1e9)
            max_next = tf.reduce_max(q_next, axis=1)
            target = R + self.discount_factor * D * max_next

            q_curr = self.model(S, training=True)
            sel = tf.gather_nd(q_curr, tf.stack([tf.range(tf.shape(A)[0]), A], axis=1))
            loss = tf.reduce_mean(tf.square(target - sel))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if random.random() < 1e-4:
            tf.keras.backend.clear_session()

    # exploration helpers
    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def set_exploration_rate(self, new_er):
        self.exploration_rate = new_er
