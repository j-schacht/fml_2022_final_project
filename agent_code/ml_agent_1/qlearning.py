from collections import namedtuple
import numpy as np

from agent_code.ml_agent_1.train import Transition

Transition = namedtuple('Transition', ('X', 'action', 'nextX', 'reward'))
I_X = 0
I_ACTION = 1
I_NEXTX = 2
I_REWARD = 3

class QLearningModel:
    """ 
    This class implements the Q-Learning model as a method for reinforcement learning,
    optimized for large state spaces.

    update strategy: n-step q-learning, offline update using an experience buffer
    q-function regression: linear value optimization

    attribute: beta
        matrix consisting of all beta-vectors (one vector per action).
        vector length equals number of features.

    attribute: alpha
        training rate (hyperparameter)

    attribute: gamma
        discount factor (hyperparameter)
    """

    def __init__(self, num_features, num_actions, alpha, gamma, buffer_size, batch_size, initial_beta = None):

        if initial_beta is not None:
            assert type(initial_beta) is np.ndarray
            assert initial_beta.shape[0] == num_actions
            assert initial_beta.shape[1] == num_features
            self.beta = initial_beta.copy
        else:
            initial_beta = np.zeros((num_actions, num_features))

        assert batch_size < buffer_size

        self.alpha = alpha
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = np.zeros(buffer_size, dtype=Transition)

    def bufferAddTransition(self, transition):
        assert type(transition) is Transition
        assert type(transition.X) is np.ndarray and transition.X.shape == (self.num_features,)
        assert type(transition.nextX) is np.ndarray and transition.nextX.shape == (self.num_features,)
        assert type(transition.action) is int
        assert type(transition.reward) is int

        np.roll(self.buffer, 1, axis=0)
        self.buffer[0] = transition

    def gradientUpdate(self):
        selection = np.zeros((self.num_actions, self.batch_size), dtype=int)
        for i in range(self.num_actions):
            selection[i] = np.random.choice(self.buffer_size, size=self.batch_size, replace=False)

        X = self.buffer[selection][:,:,I_X]             # dim: (num_actions x batch_size x num_features)
        nextX = self.buffer[selection][:,:,I_NEXTX]     # dim: (num_actions x batch_size x num_features)
        reward = self.buffer[selection][:,:,I_REWARD]   # dim: (num_actions x batch_size x 1)

        # find maximum value of Q for nextX and any possible action
        # dim:  max((num_actions x batch_size x num_features) * (num_actions x num_features)^T, axis=2)
        #     = max((num_actions x batch_size x num_actions), axis=2)
        #     = (num_actions x batch_size x 1) 
        maxQ = np.max(np.matmul(nextX, self.beta.T), axis=2)

        # calculate expected response Y
        # dim:  (num_actions x batch_size x 1) + ((1x1) * (num_actions x batch_size x 1))
        #     = (num_actions x batch_size x 1)
        Y = reward + (self.gamma * maxQ)

        # calculate the new beta-vectors (= gradient update)
        # dim:  (num_actions x num_features) + (1x1) * 
        # (batch_size x num_features)^T * ((batch_size x 1) - (batch_size x num_features) * (num_features x 1)) 
        # (num_features x batch_size)   * ((batch_size x 1) - (batch_size x 1))
        # (num_features x 1)
        self.beta[a] = self.beta[a] + (self.alpha / self.batch_size) * np.sum(X[a] * (Y[a] - X[a] * self.beta[a]))
        

    def Q(self, X, a):
        assert type(X) is np.ndarray and X.shape == (self.num_features,)
        assert a < self.num_actions and a >= 0
        return np.matmul(X, self.beta[a])