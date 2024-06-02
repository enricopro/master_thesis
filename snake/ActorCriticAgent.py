import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.layers import Flatten
from keras.models import Model

class ActorCriticAgent:
    def __init__(self, action_space_size, state_space_shape, actor_learning_rate=0.001, critic_learning_rate=0.01, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        self.action_space_size = action_space_size
        self.state_space_shape = state_space_shape
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.actor, self.critic = self.build_models()

    def build_models(self):
        actor_model = Sequential([
            Flatten(input_shape=self.state_space_shape[1:]),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_space_size, activation='softmax')
        ])
        actor_model.compile(optimizer=Adam(lr=self.actor_learning_rate), loss='categorical_crossentropy')

        critic_model = Sequential([
            Flatten(input_shape=self.state_space_shape[1:]),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        critic_model.compile(optimizer=Adam(lr=self.critic_learning_rate), loss='mse')

        return actor_model, critic_model

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return [[np.random.randint(self.action_space_size)] for _ in range(self.state_space_shape[0])]
        else:
            action_probs = self.actor.predict(state)
            actions = [np.random.choice(self.action_space_size, p=probs) for probs in action_probs]
            return [[action] for action in actions]

    def update_actor_critic(self, state, action, reward, next_state):
        next_state_value = self.critic.predict(next_state)
        target = reward + self.discount_factor * next_state_value #shape 1000,1
        
        advantages = target - self.critic.predict(state)
        action_masks = np.eye(self.action_space_size)[np.array(action).reshape(-1)]
        
        self.actor.fit(state, action_masks * advantages, verbose=0)
        self.critic.fit(state, target, verbose=0)

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
