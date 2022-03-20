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

    attribute: num_features
        the number of features (= dimension of the 1D feature vector).

    attribute: num_actions
        the number of possible actions.

    attribute: beta
        matrix consisting of all beta-vectors (one vector per action).
        vector length equals number of features.

    attribute: alpha
        learning rate (hyperparameter)

    attribute: gamma
        discount factor (impact of future reward).
        (hyperparameter)

    attribute: buffer_size
        size of the experience buffer (= number of transitions to remember & consider).
        (hyperparameter)

    attribute: batch_size
        size of the random samples taken from the experience buffer to do the gradient update.
        this must be smaller than buffer_size.
        (hyperparameter)

    attribute: buffer
        this is the experience buffer storing transitions. size is given by buffer_size.
    """

    def __init__(self, num_features, num_actions, alpha, gamma, buffer_size, batch_size, initial_beta = None):
        """
        Initialization of the QLearningModel.

        :param num_features: number of features [int]
        :param num_actions: number of actions [int]
        :param alpha: training rate [0.0 <= alpha <= 1.0] [float]
        :param gamma: discount factor [0.0 <= gamma <= 1.0] [float]
        :param buffer_size: size of the experience buffer [int]
        :param batch_size: size of random samples from experience buffer [0 < batch_size < buffer_size] [int]
        :param initial_beta: initial values for the beta vectors (default is zero) [np.ndarray]
        """
        assert type(num_features) is int and num_features > 0
        assert type(num_actions) is int and num_actions > 0
        assert type(alpha) is float and alpha >= 0.0 and alpha <= 1.0
        assert type(gamma) is float and gamma >= 0.0 and gamma <= 1.0
        assert type(buffer_size) is int and buffer_size > batch_size
        assert type(batch_size) is int and batch_size > 0

        if initial_beta is not None:
            assert type(initial_beta) is np.ndarray
            assert initial_beta.shape == (num_actions, num_features)
            self.beta = initial_beta.copy()
        else:
            initial_beta = np.zeros((num_actions, num_features))

        self.num_features = num_features
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = np.zeros(buffer_size, dtype=Transition)


    def bufferAddTransition(self, transition):
        """
        Add transition to experience buffer.

        :param transition: tuple of (X, action, nextX, reward)
            X: 1D feature vector of state t [np.ndarray]
            action: integer representing the action [int]
            nextX: 1D feature vector of state t+1 [np.ndarray]
            reward: absolute reward gotten after the transition [int]
        """
        assert type(transition) is Transition
        assert type(transition.X) is np.ndarray and transition.X.shape == (self.num_features,)
        assert type(transition.nextX) is np.ndarray and transition.nextX.shape == (self.num_features,)
        assert type(transition.action) is int
        assert type(transition.reward) is int

        np.roll(self.buffer, 1, axis=0)
        self.buffer[0] = transition


    def bufferClear(self):
        """
        Clear experience buffer.
        """
        self.buffer = np.zeros(self.buffer_size, dtype=Transition)


    def gradientUpdate(self):
        """
        Compute new gradients by considering the transitions from the experience buffer.
        Lecture reference: pp. 159-162
        """
        # generate a batch (= random subset of the experience buffer) for each beta-vector
        selection = np.zeros((self.num_actions, self.batch_size), dtype=int)
        for i in range(self.num_actions):
            selection[i] = np.random.choice(self.buffer_size, size=self.batch_size, replace=False)

        X = self.buffer[selection][:,:,I_X]             # dim: (num_actions x batch_size x num_features)
        nextX = self.buffer[selection][:,:,I_NEXTX]     # dim: (num_actions x batch_size x num_features)
        reward = self.buffer[selection][:,:,I_REWARD]   # dim: (num_actions x batch_size x 1)

        # find maximum value of Q for nextX and any possible action
        # lecture reference: p. 160
        # dim:  max((num_actions x batch_size x num_features) * (num_actions x num_features)^T, axis=2)
        #     = max((num_actions x batch_size x num_actions), axis=2)
        #     = (num_actions x batch_size x 1) 
        maxQ = np.max(np.matmul(nextX, self.beta.T), axis=2)

        # calculate expected response Y
        # # lecture reference: p. 160
        # dim:  (num_actions x batch_size x 1) + ((1x1) * (num_actions x batch_size x 1))
        #     = (num_actions x batch_size x 1)
        Y = reward + (self.gamma * maxQ)

        # calculate the new beta-vectors (= gradient update)
        for i in range(self.num_actions):
            # lecture reference: p. 162
            # dim:  (num_features x 1) + (1x1) * sum(((batch_size x num_features)^T . ((batch_size x 1) - (batch_size x num_features) * (num_features x 1)))^T)
            #     = (num_features x 1) + (1x1) * sum(((batch_size x num_features)^T .  (batch_size x 1))^T)
            #     = (num_features x 1) + (1x1) * sum((batch_size x num_features), axis=0)
            #     = (num_features x 1) + (1x1) * (num_features x 1)
            #     = (num_features x 1)
            self.beta[i] = self.beta[i] + (self.alpha / self.batch_size) * np.sum((X[i].T * (Y[i] - np.matmul(X[i] * self.beta[i]))).T)
        

    def Q(self, X, a):
        """
        This is the action value function. It returns a value for a given combination of a 1D feature vector and 
        an action. It can be used to find the best action for the current game state.
        """
        assert type(X) is np.ndarray and X.shape == (self.num_features,)
        assert a < self.num_actions and a >= 0
        return np.matmul(X, self.beta[a])


    def predictAction(self, X):
        """
        Based on what the model has learned so far, this function returns the best action for a given 
        1D feature vector representing the game state.
        This function is basically just calculating the return of the Q-function for every possible action.
        """
        assert type(X) is np.ndarray and X.shape == (self.num_features,)
        return np.argmax(np.matmul(X, self.beta.T))

    
    