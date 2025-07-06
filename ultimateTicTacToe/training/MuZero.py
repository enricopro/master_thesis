import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

class MuZero:
    def __init__(self, raw_states, input_dim, action_dim, player, learning_rate=0.0003, simulations=500, hidden_size=128, loaded=False):
        self.simulations = simulations
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        if loaded:
            self.representation_model = models.load_model('./models/muzero/representation_model')
            self.dynamics_model = models.load_model('./models/muzero/dynamics_model')
            self.value_model = models.load_model('./models/muzero/value_model')
            self.policy_model = models.load_model('./models/muzero/policy_model')
        else:
            self.representation_model = self.build_representation_model()
            self.dynamics_model = self.build_dynamics_model()
            self.value_model = self.build_value_model()
            self.policy_model = self.build_policy_model()
        self.roots = [MuZeroNode(raw_state, player) for raw_state in self.predict_representation(raw_states)]

    def build_representation_model(self):
        inputs = layers.Input(shape=(self.input_dim,))  # Input is a vector of shape (81,)
        x = layers.Dense(self.hidden_size, activation='relu')(inputs)
        x = layers.Dense(self.hidden_size, activation='relu')(x)
        model = models.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def build_dynamics_model(self):
        state_input = layers.Input(shape=(self.hidden_size,))
        action_input = layers.Input(shape=(self.action_dim,))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(self.hidden_size, activation='relu')(x)
        next_state = layers.Dense(self.hidden_size)(x)  # Predict the next latent state
        reward = layers.Dense(1)(x)  # Predict the reward
        model = models.Model(inputs=[state_input, action_input], outputs=[next_state, reward])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def build_value_model(self):
        state_input = layers.Input(shape=(self.hidden_size,))
        x = layers.Dense(self.hidden_size, activation='relu')(state_input)
        value = layers.Dense(1)(x)  # Predict the value of the latent state
        model = models.Model(inputs=state_input, outputs=value)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def build_policy_model(self):
        state_input = layers.Input(shape=(self.hidden_size,))
        x = layers.Dense(self.hidden_size, activation='relu')(state_input)
        policy = layers.Dense(self.action_dim, activation='softmax')(x)  # Predict action probabilities
        model = models.Model(inputs=state_input, outputs=policy)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        return model

    def predict_representation(self, raw_state):
        return self.representation_model.predict(raw_state, verbose=0)

    def predict_policy(self, latent_state):
        return self.policy_model.predict(latent_state, verbose=0)

    def predict_value(self, latent_state):
        return self.value_model.predict(latent_state, verbose=0)

    def predict_dynamics(self, latent_state, action):
        latent_state = np.array(latent_state)
        action = np.array(action)
        return self.dynamics_model.predict([latent_state, action], verbose=0)

    def simulate(self, available_actions):
        for _ in range(self.simulations):
            nodes_to_expand = np.array([])
            # For each game tree we reach the leaves
            for index, root in enumerate(self.roots):
                current_node = root
                # First of all we reach the leaves and we append them to 'nodes_to_expand'
                while True:
                    if len(current_node.children) == 0:
                        nodes_to_expand = np.append(nodes_to_expand, current_node)
                        break
                    else:
                        current_node = current_node.select_child()

            # # I explore up to the maximum depth of the tree
            # for _ in range(self.action_dim):
            # We call the policy model
            latent_states_nodes_to_expand = np.array([n.state for n in nodes_to_expand])
            policies = self.predict_policy(latent_states_nodes_to_expand)
            values = self.predict_value(latent_states_nodes_to_expand)

            # For each value, we add the value into the node
            for index, node in enumerate(nodes_to_expand):
                node.value = values[index]

            # Do a sampling of for each policy
            sampled_actions = np.array([np.random.choice(policy.shape[0], p=policy) for policy in policies])
            
            # Add children with that action
            for index, node in enumerate(nodes_to_expand):
                node.expand(policies[index])

            # Now, we call the dynamics model
            child_latent_states, rewards = self.predict_dynamics(latent_states_nodes_to_expand, sampled_actions)
            values = self.predict_value(child_latent_states)
            for index, node in enumerate(nodes_to_expand):
                node.children[sampled_actions[index]].state = child_latent_states[index]
                node.children[sampled_actions[index]].reward = rewards[index]
                nodes_to_expand[index] = node

            # Backpropagation
            dones_propagation = np.zeros(len(self.roots))

            # while np.sum(dones_propagation) != len(self.roots):
            #     values = self.predict_value(child_latent_states)
            #     for index, node in enumerate(nodes_to_expand):
            #         if node.parent is None:
            #             dones_propagation[index] = 1
            #             continue
            #         if node.player == self.roots[index].player:
            #             values[index] = -values[index]
            #         node.value = node.reward + 0.9 * values[index]
            #         node.visits += 1
            #         child_latent_states[index] = node.parent.state
            #         nodes_to_expand[index] = node.parent

            while np.sum(dones_propagation) != len(self.roots):
                # values = self.predict_value(child_latent_states)
                for index, node in enumerate(nodes_to_expand):
                    if node.parent is None:
                        dones_propagation[index] = 1
                        continue
                    if node.player == self.roots[index].player:
                        node.value = node.reward + 0.9 * (-values[index])
                    else:
                        node.value = node.reward + 0.9 * values[index]
                    node.visits += 1
                    child_latent_states[index] = node.parent.state
                    nodes_to_expand[index] = node.parent

                        
        distributions = [root.get_distribution() for root in self.roots]
        distributions = np.array(distributions) + 0.0000001
        distributions = np.multiply(distributions, available_actions)
        distributions = distributions / np.sum(distributions, axis=1, keepdims=True)
        return distributions

    def train(self, states, actions, rewards, next_states):

        # Predict current and next latent states
        latent_states = self.predict_representation(states)
        next_latent_states = self.predict_representation(next_states)

        # Compute target values (e.g., bootstrap target with value model)
        target_values = rewards + 0.9 * self.predict_value(next_latent_states).flatten()

        # Train representation model on state predictions
        self.representation_model.train_on_batch(states, latent_states)

        # Train value model on value predictions
        self.value_model.train_on_batch(latent_states, target_values)

        # Train dynamics model on next state and reward predictions
        self.dynamics_model.train_on_batch([latent_states, actions], [next_latent_states, rewards])

        # Train policy model based on latent states and target actions
        self.policy_model.train_on_batch(latent_states, actions)
    
    def save_models(self, path):
        self.representation_model.save(path + '/representation_model')
        self.dynamics_model.save(path + '/dynamics_model')
        self.value_model.save(path + '/value_model')
        self.policy_model.save(path + '/policy_model')


class MuZeroNode:
    def __init__(self, latent_state=None, player=None, action=None, parent=None, policy=None, value=None):
        self.state = latent_state  # Latent state is obtained from the representation model
        self.action = action
        self.parent = parent
        self.player = player
        self.children = []
        self.value = value
        self.policy = policy if policy is not None else 0  # Policy probability
        self.visits = 0

    def expand(self, policy):
        for index, prob in enumerate(policy):
            if prob != 0.0:
                child_node = MuZeroNode(action=index, parent=self, policy=prob)
                self.children.append(child_node)
        return self.children[0]  # Return the first child (or adjust as needed)

    def select_child(self):
        total_visits = sum(child.visits for child in self.children)
        # Find the child with the maximum PUCT value
        best_child = max(self.children, key=lambda child: child.puct(total_visits))
        return best_child

    def get_distribution(self):
        total_visits = sum(child.visits for child in self.children)
        policy = np.zeros(81)
        for index, ch in enumerate(self.children):
            policy[index] = ch.puct(total_visits)
        return policy / policy.sum()

    def puct(self, total_visits):

        if self.visits == 0:
            return float('inf')  # Infinitely explore unvisited nodes

        # Constant        
        c_p = np.sqrt(2)
        # Value from the node
        Q = self.value
        # Policy probability of the action
        P = self.policy
        # Total visits of the parent node
        N = total_visits
        # Visits to this child node
        n = self.visits
        
        # Compute the exploration bonus
        U = c_p * P * (N ** 0.5) / (1 + n)
        
        # PUCT value
        return Q + U