import numpy as np
import random
from collections import deque

class CognitiveAgent:
    def __init__(self, actions, memory_size=100, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.actions = actions  # Possible decisions
        self.q_table = {}  # Stores learned schemas
        self.memory = deque(maxlen=memory_size)  # Short-term memory
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate

    def encode_state(self, state):
        # Converts complex input into a simplified 'schema' key
        return tuple(state.items())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, state):
        # Epsilon-greedy strategy: sometimes explore, sometimes exploit
        encoded_state = self.encode_state(state)
        if random.uniform(0, 1) < self.epsilon or encoded_state not in self.q_table:
            return random.choice(self.actions)
        return max(self.q_table[encoded_state], key=self.q_table[encoded_state].get)

    def update_q_table(self, state, action, reward, next_state):
        encoded_state = self.encode_state(state)
        encoded_next = self.encode_state(next_state)

        if encoded_state not in self.q_table:
            self.q_table[encoded_state] = {a: 0.0 for a in self.actions}
        if encoded_next not in self.q_table:
            self.q_table[encoded_next] = {a: 0.0 for a in self.actions}

        old_value = self.q_table[encoded_state][action]
        future_reward = max(self.q_table[encoded_next].values())

        new_value = old_value + self.alpha * (reward + self.gamma * future_reward - old_value)
        self.q_table[encoded_state][action] = new_value

    def train(self):
        for state, action, reward, next_state in self.memory:
            self.update_q_table(state, action, reward, next_state)


# Simulating the environment
env_states = [
    {"situation": "danger", "experience": "pain"},
    {"situation": "opportunity", "experience": "success"},
    {"situation": "neutral", "experience": "unknown"},
]

actions = ["avoid", "analyze", "approach", "observe"]

# Create agent
agent = CognitiveAgent(actions)

# Training simulation
for episode in range(100):
    state = random.choice(env_states)
    action = agent.choose_action(state)

    # Define simple environment reward rules
    if state["situation"] == "danger" and action == "avoid":
        reward = 10
    elif state["situation"] == "opportunity" and action == "approach":
        reward = 10
    elif state["situation"] == "neutral" and action == "observe":
        reward = 5
    else:
        reward = -5

    next_state = random.choice(env_states)
    agent.remember(state, action, reward, next_state)
    agent.train()

# Test phase
test_state = {"situation": "danger", "experience": "pain"}
decision = agent.choose_action(test_state)
print("Agent Decision in 'danger':", decision)

test_state = {"situation": "opportunity", "experience": "success"}
decision = agent.choose_action(test_state)
print("Agent Decision in 'opportunity':", decision)
