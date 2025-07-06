import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow import keras
import math

class AlphaZero:
    def __init__(self, environments, input_dim, action_dim, learning_rate=0.0003, simulations=100, actor_path='./models/Thesis_PPO/actor', critic_path='./models/Thesis_PPO/critic'):
        self.simulations = simulations
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.policy_model = self.load_policy_model(actor_path)
        self.value_model = self.load_value_model(critic_path)
        self.roots = [AlphaZeroNode(env.clone()) for env in environments]

    def load_policy_model(self, actor_path):
        model = keras.models.load_model(actor_path)  # Load the entire model
        return model

    def load_value_model(self, critic_path):
        model = keras.models.load_model(critic_path)  # Load the entire model
        return model

    def predict_policy(self, state):
        # This method predicts only the action probabilities
        states_tensor = tf.convert_to_tensor(state)
        return self.policy_model.predict(states_tensor, verbose=0)

    def predict_value(self, state):
        # This method predicts only the value of the state
        states_tensor = tf.convert_to_tensor(state)
        return self.value_model.predict(states_tensor, verbose=0)

    def play(self):
        # Run MCTS from the current state of the environment
        actions = self.simulate()
        return actions

    def simulate(self):
        for _ in range(self.simulations):
            nodes_to_expand = np.array([])
            for index, root in enumerate(self.roots):
                current_node = root
                # First of all we reach the leaves and we append them to 'nodes_to_expand'
                while True:
                    if len(current_node.children) == 0:
                        nodes_to_expand = np.append(nodes_to_expand, current_node)
                        break
                    else:
                        current_node = current_node.select_child()
            for n in nodes_to_expand:
                if n.env is None:
                    n.env = n.parent.env.clone()
                    state = n.env.step(n.action)[0]
                    n.untried_actions = n.env.get_valid_actions()
                    n.state = state.flatten()

            # Now, we predict the policy probabilities for each child
            states_nodes_to_expand = np.array([n.state for n in nodes_to_expand])

            policy_probabilities = self.predict_policy(states_nodes_to_expand)
            nodes_for_rollout = np.array([])
            for index, node_to_expand in enumerate(nodes_to_expand):
                nodes_for_rollout = np.append(nodes_for_rollout, node_to_expand.expand(policy_probabilities[index]))
            for n in nodes_for_rollout:
                if n.env is None:
                    n.env = n.parent.env.clone()
                    n.state = (n.env.step(n.action)[0]).flatten()
                    n.untried_actions = n.env.get_valid_actions()

            # Now that we selected the node to do the rollout from, we can use the value network to predict the outcome
            states_nodes_for_rollout = np.array([n.state for n in nodes_for_rollout])
            values = self.predict_value(states_nodes_for_rollout)

            # We backpropagate the values until we reach the root
            for i in range(len(nodes_for_rollout)):
                node = nodes_for_rollout[i]
                value = values[i]
                player = nodes_for_rollout[i].env.current_player
                while node:
                    node.update(value, player)
                    node = node.parent

        # Now we have done all the simulations, we can choose the best action for each environment
        best_actions = []
        for root in self.roots:
            best_actions.append(root.best_child().action)
        return best_actions

    def update_tree_with_move(self, actions):
        for index, root in enumerate(self.roots):
            # Find the child node with the given action and make it the new root
            for ch in root.children:
                if actions[index] == ch.action:
                    child = ch
                    break
            if child is not None:
                if child.env is None:
                    child.env = child.parent.env.clone()
                    state = child.env.step(child.action)[0]
                    child.state = state.flatten()
                    child.untried_actions = child.env.get_valid_actions()
                self.roots[index] = child
                self.roots[index].parent = None  # Detach the new root from its parent

                # Since we've moved to a new root, reflect this in win/loss values for the game perspective
                self.roots[index].change_values_for_player()

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


class AlphaZeroNode:
    def __init__(self, env, policy=0, parent=None, action=None):
        self.env = env
        self.state = None  # Game state at this node
        self.parent = parent  # Parent node
        self.action = action  # Action leading to this state
        self.children = []  # List of child nodes
        self.wins = 0  # Number of wins after this node
        self.visits = 0  # Number of visits to this node
        self.sum_of_children_visits = 0  # Number of visits of children
        self.untried_actions = []
        self.policy = policy
        if env is not None:
            state, self.untried_actions = env.to_state()  # Game state at this node
            self.state = state.flatten()

    def puct(self, exploration_constant=1.414): # 1.414 = sqrt(2)
        # Calculate the PUCT value used for node selection
        if self.visits == 0:
            return float('inf')  # Avoid division by zero
        return (self.wins / self.visits + exploration_constant * self.policy * math.sqrt(self.sum_of_children_visits) / (1 + self.visits))[0]

    def expand(self, policy_probabilities):

        # Normalize probabilities
        policy_probabilities = policy_probabilities + 0.0000001
        policy_probabilities = policy_probabilities * self.env.get_valid_actions()
        policy_probabilities = policy_probabilities / policy_probabilities.sum()

        valid_actions_indices = np.where(self.untried_actions == 1)[0]

        # Initialize all child nodes for the valid actions
        for index in valid_actions_indices:
            child_node = AlphaZeroNode(None, parent=self, action=index, policy=policy_probabilities[index])
            self.children.append(child_node)

        # Update untried actions to reflect that these actions have now been initialized
        self.untried_actions[valid_actions_indices] = 0

        # Sample a random action
        sampled_index = np.random.choice(len(policy_probabilities), p=policy_probabilities)

        # Get the result as a numpy value
        sampled_index = tf.squeeze(sampled_index).numpy()

        # Return the corresponding child node
        for ch in self.children:
            if sampled_index == ch.action:
                return ch

    def update(self, result, last_move_player):
        # Update this node's statistics
        self.visits += 1
        if last_move_player == self.env.current_player:  # Adjust according to who this node represents
            result = -result   
        self.wins += result
        if self.parent:
            self.parent.sum_of_children_visits += 1  # Update parent's sum of visits when this node is visited

    def best_child(self):
        return max(self.children, key=lambda x: x.visits)

    def select_child(self):
        return max(self.children, key=lambda node: node.puct())

    def change_values_for_player(self):
        # Negate the wins to reflect the change in perspective
        self.wins = -self.wins

        # Recursively change the values for all children nodes
        for child in self.children:
            child.change_values_for_player()

    def clone(self):
        # Create a new instance with a copied state
        cloned_node = AlphaZeroNode(self.env.clone(), None, self.action, policy=self.policy)
        cloned_node.children = self.children
        cloned_node.wins = self.wins
        cloned_node.visits = self.visits
        cloned_node.untried_actions = list(self.untried_actions) if self.untried_actions else None
        return cloned_node
