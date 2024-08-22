import numpy as np
import math
from random import choice
from UltimateTicTacToeEnvSelfPlay import UltimateTicTacToeEnvSelfPlay

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Game state at this node
        self.parent = parent  # Parent node
        self.action = action  # Action leading to this state
        self.children = []  # List of child nodes
        self.wins = 0  # Number of wins after this node
        self.visits = 0  # Number of visits to this node
        self.untried_actions = state.get_valid_actions()  # Actions not yet 

    def uct(self, exploration_constant=1.414):
        # Calculate the uct value used for node selection
        if self.visits == 0:
            return float('inf')  # Avoid division by zero
        return self.wins / self.visits + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

    def expand(self):
        # Expand the tree by creating a new child node
        for index, action in enumerate(self.untried_actions):
            if action == 1:
                next_state = self.state.clone()
                next_state.step(index)
                child_node = MCTSNode(next_state, parent=self, action=index)
                self.children.append(child_node)
        return choice(self.children)

    def update(self, result, last_move_player):
        # Update this node's statistics
        self.visits += 1
        if last_move_player == self.state.current_player:  # Adjust according to who this node represents
            self.wins += 1
        # if last_move_player != self.state.current_player:  # Adjust according to who this node represents
        #     result = -result
        # self.wins += result

    def best_child(self):
        # Select the child with the highest number of visits
        # print("###############")
        # for ch in self.children:
        #     print(ch.action, ": ", ch.visits)
        return max(self.children, key=lambda x: x.visits)

    def select_child(self):
        # Select child with highest uct value
        # print("###############")
        # for ch in self.children:
        #     print(ch, ": ", ch.uct())
        return max(self.children, key=lambda x: x.uct())

class MCTS:
    def __init__(self, environment, simulations=1000):
        self.environment = environment
        self.simulations = simulations

    def simulate(self):
        # Simulate a game from the current state
        current_node = MCTSNode(self.environment.clone())
        for _ in range(self.simulations):
            node = current_node
            while True:
                if not node.children:
                    if node.visits == 0:
                        result, player = self.rollout(node.state.clone())
                        break
                    else:
                        node = node.expand()
                        result, player = self.rollout(node.state.clone())
                        break
                else:
                    node = node.select_child()
            while node:
                node.update(result, player)
                node = node.parent
        # self.environment.render()
        # for ch in current_node.children:
        #     print(ch.action, ": ", ch.wins)
        # print("BEST ACTION: ", current_node.best_child().action)
        return current_node.best_child().action

    def rollout(self, state):
        # Perform a random rollout from the given state
        while True:
            valid_actions = state.get_valid_actions()
            if not valid_actions.any():
                break
            action = choice([a for a in range(len(valid_actions)) if valid_actions[a] > 0])
            current_player = state.current_player
            _, reward, done, _ = state.step(action)
            if done:
                return reward, current_player
        return 0, state.current_player  # Game ended in a draw

    def play(self):
        # Run MCTS from the current state of the environment
        action = self.simulate()
        return action
    
    def update_tree_with_move(self, action):
        # Find the child node with the given action and make it the new root
        for child in self.root.children:
            if child.action == action:
                self.root = child
                self.root.parent = None  # Detach the new root from its parent
                return
        # If no child with the action is found, reset the tree
        self.root = MCTSNode(self.environment.clone())
