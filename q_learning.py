import numpy as np
import gym
import pickle as pkl

cliffEnv = gym.make("CliffWalking-v0")
q_table = np.zeros(shape=(48, 4))


def policy(state, explore=0.0):
    next_action = int(np.argmax(q_table[state]))
    if (np.random.random() <= explore):
        next_action = int(np.random.randint(low=1, high=4, size=1))

    return next_action


EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
EPISODES = 500

for episode in range(EPISODES):
    state = cliffEnv.reset()
    action = policy(state, EPSILON)
    done = False
    total_reward = 0
    episode_len = 0

    while not done:
        next_state, reward, done, _ = cliffEnv.step(action)
        next_action = policy(next_state)

        q_table[state][action] += ALPHA*(reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])
        state = next_state
        action = next_action
        total_reward += reward
        episode_len += 1
    print("Episodes: ", episode, "NumEpisodes: ", episode_len, "Total Reward: ", total_reward)

cliffEnv.close()
pkl.dump(q_table, open("q_learning_q_table.pkl", "wb"))
print("Training complete. pickle table dumped...")

