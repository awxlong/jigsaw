import numpy as np
import random

class RLPuzzleAgent:
    def __init__(self, grid_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.8, exploration_decay=0.995, min_exploration_rate=0.01):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = {}

    def get_state_key(self, state):
        """Convert the state (a list) to a tuple so it can be used as a dictionary key."""
        return tuple(state)

    def choose_action(self, state, possible_actions):
        """Choose an action using epsilon-greedy strategy."""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(possible_actions))

        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: choose a random action index
            return random.choice(range(len(possible_actions)))
        else:
            # Exploit: choose the action with the highest Q-value
            return np.argmax(self.q_table[state_key])

    def update_q_table(self, state, action_index, reward, next_state):
        """Update the Q-table using the Q-learning formula."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.get_possible_actions(state)))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.get_possible_actions(next_state)))

        # Q-learning formula
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.discount_factor * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action_index]
        self.q_table[state_key][action_index] += self.learning_rate * td_error

    def decay_exploration_rate(self):
        """Decay the exploration rate over time."""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def get_possible_actions(self, state):
        """Get possible actions (moves) for the current state."""
        empty_pos = state.index(self.grid_size * self.grid_size - 1)
        row, col = empty_pos // self.grid_size, empty_pos % self.grid_size
        possible_actions = []

        if row > 0:
            possible_actions.append(empty_pos - self.grid_size)  # Move up
        if row < self.grid_size - 1:
            possible_actions.append(empty_pos + self.grid_size)  # Move down
        if col > 0:
            possible_actions.append(empty_pos - 1)  # Move left
        if col < self.grid_size - 1:
            possible_actions.append(empty_pos + 1)  # Move right

        return possible_actions