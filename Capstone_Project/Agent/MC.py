import sys
import numpy as np
from Environment import Easy21
from helper import epsilon_greedy


def mc_control(N0=100, num_episodes=1000):
    """

    :param N0:integer constant
    :param num_episodes:Positive integer
    :return: Q & V
    """
    # Get the environment
    env = Easy21()  # instantiate the env

    # initializing Zeros arrays for num of state visited and state action pair
    NS = np.zeros((11, 22))
    NSA = np.zeros((2, 11, 22))

    # Initializing state action function
    Q = np.zeros((2, 11, 22))

    # looping over episodes
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = []
        state = env.reset()
        episode.append(state)

        Gt = 0

        while True:
            # Update epsilon values
            NS[state[0], state[1]] += 1
            epsilon = N0 / (N0 + NS[state[0], state[1]])

            action = epsilon_greedy(Q, state, epsilon)
            episode.append(action)

            # Update alpha values
            NSA[action, state[0], state[1]] += 1
            alpha = 1 / NSA[action, state[0], state[1]]

            state, reward, done = env.step(action)

            # Sum the reward
            Gt += reward

            if done:
                break
            else:
                # Append state to the episode
                episode.append(state)

                # Update all states
        for index, event in enumerate(episode):
            if index % 2 == 0:
                state = event
            else:
                action = event
                Q[action, state[0], state[1]] += alpha * (Gt - Q[action, state[0], state[1]])

    return Q[:, 1:11, 1:22], np.max(Q[:, 1:11, 1:22], axis=0)