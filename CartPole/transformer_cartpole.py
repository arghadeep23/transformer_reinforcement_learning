import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from collections import deque
import random


# Transformer block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head attention
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed-forward network
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

# Transformer model
def create_transformer_model(input_shape, num_actions, sequence_length,
                             head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4,
                             mlp_units=[128, 64], dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=(sequence_length, input_shape))
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(num_actions)(x)
    return keras.Model(inputs, outputs)

# Environment setup
env = gym.make("CartPole-v1")
state_shape = env.observation_space.shape[0]
num_actions = env.action_space.n
sequence_length = 10  # Number of sequential states to consider

# Parameters (using your previous values)
ALPHA = 0.001
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
GAMMA = 0.99
NUM_EPISODES = 500
BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
MEMORY_SIZE = 10000

# Create model
model = create_transformer_model(state_shape, num_actions, sequence_length)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=ALPHA), loss='mse')

# Experience replay buffer
replay_buffer = deque(maxlen=MEMORY_SIZE)

# Create target model
target_model = create_transformer_model(state_shape, num_actions, sequence_length)
target_model.set_weights(model.get_weights())


def get_action(state_sequence, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        q_values = model.predict(state_sequence[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

# Training loop
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state_sequence = deque([state] * sequence_length, maxlen=sequence_length)
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = get_action(np.array(state_sequence), EPSILON)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1

        next_state_sequence = state_sequence.copy()
        next_state_sequence.append(next_state)

        replay_buffer.append((np.array(state_sequence), action, reward, np.array(next_state_sequence), done))

        if len(replay_buffer) >= BATCH_SIZE:
            minibatch = random.sample(replay_buffer, BATCH_SIZE)
            states = np.array([transition[0] for transition in minibatch])
            actions = np.array([transition[1] for transition in minibatch])
            rewards = np.array([transition[2] for transition in minibatch])
            next_states = np.array([transition[3] for transition in minibatch])
            dones = np.array([transition[4] for transition in minibatch])

            targets = model.predict(states, verbose=0)
            next_q_values = target_model.predict(next_states, verbose=0)
            targets[range(BATCH_SIZE), actions] = rewards + GAMMA * np.max(next_q_values, axis=1) * (1 - dones)

            model.fit(states, targets, epochs=1, verbose=0)

        state_sequence = next_state_sequence

    if episode % UPDATE_TARGET_EVERY == 0:
        target_model.set_weights(model.get_weights())

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {EPSILON:.4f}")

# Save the model
model.save("transformer_cartpole_model")

# Evaluation
eval_episodes = 100
eval_rewards = []

for _ in range(eval_episodes):
    state, _ = env.reset()
    state_sequence = deque([state] * sequence_length, maxlen=sequence_length)
    done = False
    episode_reward = 0

    while not done:
        action = get_action(np.array(state_sequence), 0)  # No exploration during evaluation
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        state_sequence.append(next_state)

    eval_rewards.append(episode_reward)

print(f"Average Evaluation Reward: {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")