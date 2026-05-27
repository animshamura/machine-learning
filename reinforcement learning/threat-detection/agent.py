correct = 0
total = 0
state = env.reset()
done = False

while not done:
    action = agent.act(state, epsilon=0.0)
    label = env.labels[env.current_step]
    correct += (action == label)
    total += 1
    state, _, done, _ = env.step(action)

print(f"Accuracy: {correct/total:.2%}")
