import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from tensorflow import keras

# Load the model
loaded_model = keras.models.load_model("transformer_cartpole_model")

# Set up the environment
env = gym.make("CartPole-v1")
sequence_length = 10  # Make sure this matches what you used during training

# Parameters
num_episodes = 100
render = False  # Set to True if you want to render the environment

# Tracking rewards
rewards = []

# Run episodes
for episode in range(num_episodes):
    state, _ = env.reset()
    state_sequence = deque([state] * sequence_length, maxlen=sequence_length)
    done = False
    total_reward = 0

    while not done:
        if render:
            env.render()

        # Get action from the model
        q_values = loaded_model.predict(np.array([state_sequence]), verbose=0)
        action = np.argmax(q_values[0])

        # Take the action
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Update the state sequence
        state_sequence.append(next_state)

    rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()

# Save rewards to a file (optional)
np.savetxt('rewards.csv', rewards, delimiter=',')

# Plotting the rewards
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode for Transformer Model')
plt.savefig('cartpole_rewards.png')
plt.show()

# Plotting the moving average of rewards
window_size = 50
moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(moving_avg)
plt.xlabel('Episode')
plt.ylabel('Moving Average of Total Reward')
plt.title(f'Moving Average of Total Reward per Episode (window size = {window_size})')
plt.savefig('cartpole_moving_avg_rewards.png')
plt.show()
