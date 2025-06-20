import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

import numpy as np
import math
from tqdm import trange
from random import choice

class AlphaZeroNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Game state at this node
        self.parent = parent  # Parent node
        self.action = action  # Action leading to this state
        self.children = []  # List of child nodes
        self.wins = 0  # Number of wins after this node
        self.visits = 0  # Number of visits to this node
        if state == None:
            self.untried_actions = None
        else:
            self.untried_actions = state.get_valid_actions()  # Actions not yet 

    def uct(self, simulations, exploration_constant=1.414):
        # Calculate the uct value used for node selection
        if self.visits == 0:
            return float('inf')  # Avoid division by zero
        return self.wins / self.visits + exploration_constant * math.sqrt(math.log(simulations) / self.visits)

    def expand(self, action_probs):
        if self.state == None:
            self.state = self.parent.state.clone()
            self.state.step(self.action)
            self.untried_actions = self.state.get_valid_actions()

        # Expand the tree by creating a new child node
        valid_actions_indices = np.where(self.untried_actions == 1)[0]
        for index in valid_actions_indices:
            child_node = AlphaZeroNode(None, parent=self, action=index)
            self.children.append(child_node)
        self.untried_actions[valid_actions_indices] = 0
        # action_probs = action_probs.flatten() #TODO maybe we can remove this (to be tested)
        action_probs = action_probs * self.state.get_valid_actions()
        action_probs = action_probs / action_probs.sum()
        choosen_action = np.argmax(action_probs)

        for ch in self.children:
            if ch.action == choosen_action:
                return ch

    def update(self, result, last_move_player):
        # Update this node's statistics
        self.visits += 1
        if last_move_player == self.state.current_player:  # Adjust according to who this node represents
            result = -result   
        self.wins += result
        # if last_move_player == self.state.current_player and result == 1:
        #     self.wins += result

    def best_child(self):
        # Return the child with the maximum number of visits, considering only valid actions
        return max((child for child in self.children), key=lambda x: x.visits)

    def select_child(self, action_probs):

        # action_probs = action_probs.flatten() #TODO maybe we can remove this (to be tested)
        action_probs = action_probs * self.state.get_valid_actions()
        action_probs = action_probs / action_probs.sum()
        choosen_action = np.argmax(action_probs)

        for ch in self.children:
            if ch.action == choosen_action:
                return ch

    def get_prob_distribution(self):
        visits=np.zeros(len(self.untried_actions))
        for ch in self.children:
            visits[ch.action] = ch.visits
        visits = visits / visits.sum()
        return visits
    
    def change_values_for_player(self):
        self.wins = -self.wins
        if len(self.children) == 0:
            return
        for ch in self.children:
            ch.change_values_for_player()

    def clone(self):
        # Create a new instance with a copied state
        cloned_node = AlphaZeroNode(self.state.clone(), None, self.action)
        cloned_node.children = [child.clone() for child in self.children]
        cloned_node.wins = self.wins
        cloned_node.visits = self.visits
        cloned_node.untried_actions = list(self.untried_actions) if self.untried_actions else None
        return cloned_node


class AlphaZero:
    def __init__(self, environments, input_dim, action_dim, learning_rate=0.0003, simulations=1000):
        self.simulations = simulations
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.policy_model = self.create_policy_model(input_dim, action_dim, learning_rate)
        self.value_model = self.create_value_model(input_dim, learning_rate)
        self.current_nodes = [AlphaZeroNode(env.clone()) for env in environments]

    def create_policy_model(self, input_dim, action_dim, learning_rate):
        inputs = tf.keras.Input(shape=(input_dim,))
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.Dense(512, activation='relu')(x)
        action_probs = layers.Dense(action_dim, activation='softmax', name='action_probs')(x)
        model = tf.keras.Model(inputs=inputs, outputs=action_probs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      loss='categorical_crossentropy')
        return model

    def create_value_model(self, input_dim, learning_rate):
        inputs = tf.keras.Input(shape=(input_dim,))
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.Dense(512, activation='relu')(x)
        value = layers.Dense(1, activation='tanh', name='value')(x)
        model = tf.keras.Model(inputs=inputs, outputs=value)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      loss='mean_squared_error')
        return model

    def predict_policy(self, state):
        # This method predicts only the action probabilities
        return self.policy_model.predict(state, verbose=0)

    def predict_value(self, state):
        # This method predicts only the value of the state
        return self.value_model.predict(state, verbose=0)

    def train(self, states, actions, values):
        self.policy_model.fit(states, actions, epochs=1, verbose=0)
        self.value_model.fit(states, values, epochs=1, verbose=0)

    def simulate(self):
        nodes = []
        best_actions = []
        states = self.get_states(self.current_nodes)
        action_probs = self.predict_policy(states)
        for index, current_node in enumerate(self.current_nodes):
            # Simulate a game from the current state
            for _ in range(self.simulations):
                node = current_node
                while True:
                    if len(node.children) == 0:
                        node = node.expand(action_probs[index])
                        break
                    else:
                        node = node.select_child(action_probs[index])
                nodes.append(node)
                
        results, players = self.rollout(nodes)
        for i in range(len(results)):
            node = nodes[i]
            result = results[i]
            player = players[i]
            while node:
                node.update(result, player)
                node = node.parent
        for current_node in self.current_nodes:
            best_actions.append(current_node.best_child().action)
        return best_actions

    def rollout(self, nodes):
        # Perform a rollout from the given state using the model
        current_players = []
        for node in nodes:
            if node.state == None:
                node.state = node.parent.state.clone()
                node.state.step(node.action)
                node.untried_actions = node.state.get_valid_actions()
            current_players.append(node.state.current_player)
        states = self.get_states(nodes)
        values = self.predict_value(states)
        return values, current_players

    def play(self):
        # Run MCTS from the current state of the environment
        actions = self.simulate()
        return actions
    
    def update_tree_with_move(self, actions):
        for index, current_node in enumerate(self.current_nodes):
            # Find the child node with the given action and make it the new root
            for child in current_node.children:
                if child.action == actions[index]:
                    if child.state == None:
                        child.state = child.parent.state.clone()
                        child.state.step(child.action)
                        child.untried_actions = child.state.get_valid_actions()
                    self.current_nodes[index] = child
                    self.current_nodes[index].parent = None  # Detach the new root from its parent
                    break
            self.current_nodes[index].change_values_for_player()

    def get_prob_distribution(self):
        probs = []
        for current_node in self.current_nodes:
            probs.append(current_node.get_prob_distribution())
        return probs
    
    def get_values(self):
        values = []
        for current_node in self.current_nodes:
            values.append(current_node.uct(self.simulations))
        return values
    
    def get_states(self, nodes):
        states = []
        for node in nodes:
            states.append(node.state.to_state()[0])
        return tf.convert_to_tensor(states)

    def clone_with_new_envs(self, envs):
        # Clone both the policy and value neural network models
        cloned_policy_model = tf.keras.models.clone_model(self.policy_model)
        cloned_policy_model.set_weights(self.policy_model.get_weights())

        cloned_value_model = tf.keras.models.clone_model(self.value_model)
        cloned_value_model.set_weights(self.value_model.get_weights())

        cloned_environments = [env.clone() for env in envs]

        # Create a new instance of AlphaZero with cloned models
        cloned_instance = AlphaZero(cloned_environments, self.input_dim, self.action_dim, self.learning_rate, self.simulations)
        cloned_instance.policy_model = cloned_policy_model
        cloned_instance.value_model = cloned_value_model

        # Recreate the node structure as needed
        cloned_instance.current_nodes = [AlphaZeroNode(env) for env in cloned_environments]

        return cloned_instance


