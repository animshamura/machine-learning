import gym
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from gym import spaces

class CyberThreatEnv(gym.Env):
    def __init__(self):
        super(CyberThreatEnv, self).__init__()
        self.data, self.labels = make_classification(n_samples=10000, n_features=20, 
                                                     n_informative=15, n_classes=2, 
                                                     weights=[0.7, 0.3], random_state=42)
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

        self.current_step = 0
        self.action_space = spaces.Discrete(2)  # 0 = benign, 1 = threat
        self.observation_space = spaces.Box(low=-3, high=3, shape=(20,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action):
        label = self.labels[self.current_step]
        reward = 1 if action == label else -1
        self.current_step += 1

        done = self.current_step >= len(self.data) - 1
        next_state = self.data[self.current_step] if not done else np.zeros(20)
        return next_state, reward, done, {}
