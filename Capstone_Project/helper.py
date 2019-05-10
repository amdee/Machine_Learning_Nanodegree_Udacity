import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    :param Q:
    :param state:
    :param epsilon:
    :return: array of argmax(Q)
    """
    if np.random.rand() < epsilon:
        return np.random.randint(0, 2)
    return np.argmax(Q[:, state[0], state[1]])


def phi(state, action):
    """
    :param state: integer list of dealer, player value
    :param action: integer 0 0r 1 (stick or hit)
    :return: flatten feature
    """
    feature_vector = np.zeros((3, 6, 2))

    dealers = [1 <= state[0] <= 4, 4 <= state[0] <= 7, 7 <= state[0] <= 10]
    players = [1 <= state[1] <= 6, 4 <= state[1] <= 9, 7 <= state[1] <= 12,
               10 <= state[1] <= 15, 13 <= state[1] <= 18,
               16 <= state[1] <= 21]

    feature_vector[dealers, players, action] = 1

    return feature_vector.flatten()


def calculate_Q(weight):
    """
    :return: return Q value
    """
    Q = np.zeros((2, 11, 22))
    for dealer in range(1, 11):
        for player in range(1, 22):
            for action in range(2):
                Q[action, dealer, player] = np.sum(phi([dealer, player],
                                                       action) * weight)
    return Q
