import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow import keras
import math

class MCTS:
    def __init__(self, environments, input_dim, action_dim, simulations=100):
        self.simulations = simulations
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.roots = [MCTSNode(env.clone()) for env in environments]

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

            nodes_for_rollout = np.array([])
            for index, node_to_expand in enumerate(nodes_to_expand):
                nodes_for_rollout = np.append(nodes_for_rollout, node_to_expand.expand())
            for n in nodes_for_rollout:
                if n.env is None:
                    n.env = n.parent.env.clone()
                    n.state = (n.env.step(n.action)[0]).flatten()
                    n.untried_actions = n.env.get_valid_actions()

            values, current_players = self.rollout(nodes_for_rollout)

            # We backpropagate the values until we reach the root
            for i in range(len(nodes_for_rollout)):
                node = nodes_for_rollout[i]
                value = values[i]
                player = current_players[i]
                while node:
                    node.update(value, player)
                    node = node.parent

        # Now we have done all the simulations, we can choose the best action for each environment
        best_actions = []
        for root in self.roots:
            best_actions.append(root.best_child().action)
        return best_actions

    def rollout(self, nodes):
        # Perform a random rollout from the given state
        values = np.array([])
        current_players = np.array([])
        for node in nodes:
            if node.env == None:
                node.env = node.parent.env.clone()
                node.state = node.env.step(node.action)[0]
                node.untried_actions = node.env.get_valid_actions()
            cloned_env = node.env.clone()
            while True:
                available_actions = cloned_env.get_valid_actions()
                chosen_action = np.random.choice(np.where(available_actions == 1)[0])
                player = cloned_env.current_player
                _, reward, done, _ = cloned_env.step(chosen_action)
                if done:
                    values = np.append(values, reward)
                    current_players = np.append(current_players, player)
                    break
        return values, current_players

    def update_tree_with_move(self, actions):
        for index, root in enumerate(self.roots):
            # Find the child node with the given action and make it the new root
            child = None
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

        # Create a new instance of MCTS with cloned models
        cloned_instance = MCTS(cloned_environments, self.input_dim, self.action_dim, self.learning_rate, self.simulations)

        # Recreate the node structure as needed
        cloned_instance.roots = [MCTSNode(env) for env in cloned_environments]

        return cloned_instance


class MCTSNode:
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

    def uct(self, exploration_constant=1.414):
        # Calculate the uct value used for node selection
        if self.visits == 0:
            return float('inf')  # Avoid division by zero
        return (self.wins / self.visits) + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

    def expand(self):

        valid_actions_indices = np.where(self.untried_actions == 1)[0]

        # Initialize all child nodes for the valid actions
        for index in valid_actions_indices:
            child_node = MCTSNode(None, parent=self, action=index)
            self.children.append(child_node)

        # Update untried actions to reflect that these actions have now been initialized
        self.untried_actions[valid_actions_indices] = 0

        return self.select_child()

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
        return max(self.children, key=lambda node: node.uct())

    def change_values_for_player(self):
        # Negate the wins to reflect the change in perspective
        self.wins = -self.wins

        # Recursively change the values for all children nodes
        for child in self.children:
            child.change_values_for_player()

    def clone(self):
        # Create a new instance with a copied state
        cloned_node = MCTSNode(self.env.clone(), None, self.action)
        cloned_node.children = self.children
        cloned_node.wins = self.wins
        cloned_node.visits = self.visits
        cloned_node.untried_actions = list(self.untried_actions) if self.untried_actions else None
        return cloned_node
