env = CyberThreatEnv()
agent = DQNAgent(input_dim=20, output_dim=2)
episodes = 100
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {ep+1} - Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
