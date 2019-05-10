import sys
import numpy as np
from Environment import Easy21
from helper import phi, calculate_Q


def lfa_control(num_episodes, td_lambda, Q_MC):
    """
    :param num_episodes: Positive integer
    :param td_lambda: value between 0 to 1
    :param Q_MC:Q from MC
    :return: Q, V and mean square error
    """
    env = Easy21()  # instantiate the env
    gamma = 1
    epsilon = 0.05
    alpha = 0.01

    # Initialization
    Q = np.zeros((2, 11, 22))
    weight = np.zeros((36,))
    mean_square_error = []

    # looping through number of episodes
    for i_episode in range(1, num_episodes + 1):

        bin_feat_vector = np.zeros((36,))
        state = env.reset()  # start with a initial state
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        while True:
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 2)
            else:
                action = np.argmax([np.sum(phi(state, 0)*weight),
                                    np.sum(phi(state, 1)*weight)])

            next_state, reward, done = env.step(action)

            if np.random.rand() < epsilon:
                next_action = np.random.randint(0, 2)
            else:
                next_action = np.argmax([np.sum(phi(next_state, 0)*weight),
                                        np.sum(phi(next_state, 1)*weight)])
            cuboid = phi(state, action)
            cuboid_next = phi(next_state, next_action)

            Q_phi = np.sum(cuboid * weight)
            Q_phi_next = np.sum(cuboid_next * weight)

            td_lambda_error = reward + gamma * Q_phi_next - Q_phi
            bin_feat_vector = bin_feat_vector * td_lambda * gamma + cuboid
            delta_weight = alpha * td_lambda_error * bin_feat_vector

            weight += delta_weight

            state = next_state

            if done:
                break

        Q = calculate_Q(weight)
        mean_square_error.append(np.mean((Q_MC - Q[:, 1:11, 1:22]) ** 2))

    return Q[:, 1:11, 1:22], np.max(Q[:, 1:11, 1:22], axis=0), mean_square_error
