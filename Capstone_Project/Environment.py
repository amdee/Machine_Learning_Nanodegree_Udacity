import numpy as np


def random_generator():
    """
    This function generate a random number btw 1 & 10
    :return: integer btw 1 & 10
    """
    return np.random.randint(1, 11)


def draw_card(state, dealer_hand=False):
    """
    :param state: dealer & player value
    :param dealer_hand: boolean
    :return: state with update dealer abd player value
    """
    color = np.random.rand()
    if dealer_hand:
        # selecting color base on red (probability 1/3) or black (probability 2/3).
        if color < 1/3:
            state[0] -= random_generator()
        else:
            state[0] += random_generator()
    else:
        if color < 1/3:
            state[1] -= random_generator()
        else:
            state[1] += random_generator()
    return state


class Easy21:

    """
    Easy21 is card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over. Playing against a dealer
    with an infinite deck of cards (i.e. cards are sampled with replacement).

    Each draw from the deck results in a value between 1 and 10
    (uniformly distributed) with a color of red
    (probability 1/3) or black (probability 2/3).

    There are no aces or picture (face) cards in this game.
    At the start of the game both the player and
    the dealer draw one black card (fully observed).

    Each turn the player may either stick or hit (0, 1).
    If the player hits then she draws another card from the deck.
    If the player sticks she receives no further cards.

    The values of the player’s cards are added (black cards) or
    subtracted (red cards). If the player’s sum exceeds 21, or
    becomes less than 1, then she “goes bust” and loses the game (reward -1).

    If the player sticks then the dealer starts taking turns. The dealer always
    sticks on any sum of 17 or greater, and hits otherwise. If the dealer
    goes bust, then the player wins; otherwise, the outcome – win (reward +1),
    lose (reward -1), or draw (reward 0) – is the player with the largest sum.

    """
    def __init__(self):
        self.action = [1, 0]  # {'Hit':1, 'Stick':0}
        self.upper_limit = 21
        self.lower_limit = 1
        self.dealers_limit = 17

        self.reset()

    def random_action(self):
        return np.random.choice(self.action)

    def step(self, Action):
        """

        :param Action: integer 1 for Hit and 0 for Stick
        :return: state--> list, reward -->integer & done-->boolean
        """
        assert Action in self.action, "Action should be 1='Hit' or 0='Stick' and not {}".format(Action)
        
        if Action == self.action[0]:  # 1 Hit
            self._state = draw_card(self._state)
            if self._state[1] < self.lower_limit or self._state[1] > self.upper_limit:
                self._state = [0, 0]
                reward, done = -1, True
                return np.copy(self._state), reward, done
            else:
                reward, done = 0, False
                return np.copy(self._state), reward, done

        if Action == self.action[1]:  # 0 Stick

            while 0 < self._state[0] < self.dealers_limit:
                self._state = draw_card(self._state, True)

            # Check dealer bust
            if self._state[0] < self.lower_limit:
                self._state = [0, 0]
                reward, done = 1, True
                return np.copy(self._state), reward, done

            if self._state[0] > self.upper_limit:
                self._state = [0, 0]
                reward, done = 1, True
                return np.copy(self._state), reward, done

            # check to see if draw
            if self._state[0] == self._state[1]:
                self._state = [0, 0]
                reward, done = 0, True
                return np.copy(self._state), reward, done

            # Check to see if Loss
            if self._state[0] > self._state[1]:
                self._state = [0, 0]
                reward, done = -1, True
                return np.copy(self._state), reward, done

            # Check win
            if self._state[0] < self._state[1]:
                self._state = [0, 0]
                reward, done = 1, True
                return np.copy(self._state), reward, done

    def reset(self):
        """
        This function returns the initial state, dealer & player value.
        :return: list of integer of dealer & player value makes a state
        """
        self._state = [random_generator(), random_generator()]
        return np.copy(self._state)

    @property
    def action_space(self):
        return 0, 1
