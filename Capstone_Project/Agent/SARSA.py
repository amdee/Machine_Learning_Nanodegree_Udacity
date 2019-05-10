import sys
import numpy as np
from Environment import Easy21
from helper import epsilon_greedy


def sarsa_control(N0, num_episodes, td_lambda, Q_MC):
    """
    :param N0: constant integer
    :param num_episodes: Positive integer
    :param td_lambda: value between 0 to 1
    :param Q_MC: Q_MC:Q from MC
    :return: Q and mean square error
    """
    env = Easy21()  # instantiate the env
    NS = np.zeros((11, 22))
    NSA = np.zeros((2, 11, 22))
    Q = np.zeros((2, 11, 22))
    gamma = 1

    # Initializing  empty list of the mean square error
    mean_square_error = []

    # Episodes
    for i_episode in range(0, num_episodes):
        episode_state_action = np.zeros((2, 11, 22))
        state = env.reset()  # Initiate state
        action = np.random.randint(0, 2)  # Initiate action
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        while True:
            next_state, reward, done = env.step(action)

            # Update epsilon
            NS[state[0], state[1]] += 1
            epsilon = N0 / (N0 + NS[state[0], state[1]])

            greedy_action = epsilon_greedy(Q, next_state, epsilon)

            # making updates
            QSA = Q[greedy_action, next_state[0],
                    next_state[1]] - Q[action, state[0], state[1]]
            td_lambda_error = reward + gamma * QSA
            episode_state_action[action, state[0], state[1]] += 1

            # Update alpha
            NSA[action, state[0], state[1]] += 1
            alpha = 1 / NSA[action, state[0], state[1]]

            Q += alpha * td_lambda_error * episode_state_action
            episode_state_action *= td_lambda * gamma

            state, action = next_state, greedy_action
            if done:
                break

        mean_square_error.append(np.mean((Q_MC - Q[:, 1:11, 1:22]) ** 2))

    return Q[:, 1:11, 1:22], np.max(Q[:, 1:11, 1:22], axis=0), mean_square_error
