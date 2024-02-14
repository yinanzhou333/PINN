import numpy as np
import gym

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1')

# Q-learning parameters
num_episodes = 1000
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.2

# Initialize Q-table with zeros
num_states = env.observation_space.n
num_actions = env.action_space.n
Q_table = np.zeros((num_states, num_actions))

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Choose an action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q_table[state, :])  # Exploit

        # Take the chosen action and observe the new state and reward
        next_state, reward, done, _ = env.step(action)

        # Update Q-value using the Q-learning update rule
        Q_table[state, action] = (1 - learning_rate) * Q_table[state, action] + \
                                 learning_rate * (reward + discount_factor * np.max(Q_table[next_state, :]))

        # Move to the next state
        state = next_state

# Evaluate the trained agent
total_reward = 0
num_episodes_eval = 100

for _ in range(num_episodes_eval):
    state = env.reset()
    done = False

    while not done:
        action = np.argmax(Q_table[state, :])
        state, reward, done, _ = env.step(action)
        total_reward += reward

average_reward = total_reward / num_episodes_eval
print(f"Average reward over {num_episodes_eval} episodes: {average_reward}")

# Close the environment
env.close()

