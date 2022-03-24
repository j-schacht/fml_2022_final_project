from collections import namedtuple
import numpy as np
from os.path import exists
import os.path
from datetime import datetime
from numba import njit, prange

Transition = namedtuple('Transition', ('X', 'action', 'nextX', 'reward'))

INITIAL_BETA = np.array([[1,-0.1,-0.1,-0.1],
                        [-0.1, 1,-0.1,-0.1],
                        [-0.1,-0.1,1,-0.1],
                        [-0.1,-0.1,-0.1,1],
                        [-0.1,-0.1,-0.1,-0.1],
                        [-0.5,-0.5,-0.5,-0.5],
])

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
        initialized with normal distribution unless initial beta is given or existing model is loaded.

    attribute: beta_new 
        this is true if beta has been initialized with normal distribution, i.e. no initial beta
        was given and no existing model has been loaded from file.

    attribute: alpha
        learning rate (hyperparameter)

    attribute: gamma
        discount factor (impact of future reward).
        (hyperparameter)

    attribute: n
        parameter n for n-step q-learning (hyperparameter).

    attribute: buffer_size
        size of the experience buffer (= number of transitions to remember & consider).
        (hyperparameter)

    attribute: batch_size
        size of the random samples taken from the experience buffer to do the gradient update.
        this must be smaller than buffer_size.
        (hyperparameter)

    attribute: buffer_X, buffer_nextX, buffer_action, buffer_reward
        these are the experience buffers storing the four attributes of transitions. 
        size is given by buffer_size.

    attribute: buffer_counter
        this stores the number of elements present in the buffer

    attribute: path
        path to the file where the trained model is stored / is to be stored.

    attribute: backup_path
        path to the backup file where the trained model is stored / is to be stored. 

    attribute: training_mode
        if true, setupTraining() has been called and the model can be trained.
    """

    def __init__(self, num_features, num_actions, path="model"):
        """
        Initialization of the QLearningModel.
        If at the given path no model can be found, a new model will be created.
        
        :param num_features: number of features [int]
        :param num_actions: number of actions [int]
        :param path: path of the file to load/store the trained model without file extension [str]
        """
        assert type(num_features) is int and num_features > 0
        assert type(num_actions) is int and num_actions > 0
        assert type(path) is str and len(path) > 0

        self.num_features = num_features
        self.num_actions = num_actions
        self.path = path + ".npy"

        dir = os.path.dirname(path)
        file = os.path.basename(path)
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.backup_path = os.path.join(dir, "model_backup", file + "_" + date_time + ".npy")

        self.training_mode = False

        if exists(self.path):
            file = open(self.path, 'rb')
            self.beta = np.load(file)
            config = np.load(file)
            file.close()
            self.beta_new = False

            assert config[0] == num_features
            assert config[1] == num_actions
            assert self.beta.shape == (num_actions, num_features)

        else: 
            self.beta = np.random.uniform(low=0.0, high=1.0, size=(num_actions, num_features))
            self.beta_new = True


    def setupTraining(self, alpha, gamma, buffer_size, batch_size, n=0, initial_beta = INITIAL_BETA):
        """
        This function sets up everything needed to train the model. It needs to be called only
        if the model is to be trained.
        This function can only be called once!

        :param alpha: training rate [0.0 <= alpha <= 1.0] [float]
        :param gamma: discount factor [0.0 <= gamma <= 1.0] [float]
        :param buffer_size: size of the experience buffer [int]
        :param batch_size: size of random samples from experience buffer [0 < batch_size < buffer_size] [int]
        :param n: parameter n for n-step q-learning (nstep_gradientUpdate() can only be called if this is set larger than zero).
        :param initial_beta: initial values for the beta vectors (default is uniform distribution) [np.ndarray]
        """
        assert type(alpha) is float and alpha >= 0.0 and alpha <= 1.0
        assert type(gamma) is float and gamma >= 0.0 and gamma <= 1.0
        assert type(buffer_size) is int and buffer_size > batch_size
        assert type(batch_size) is int and batch_size > 0
        assert self.training_mode == False

        self.alpha = alpha
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n = n

        # one buffer for each attribute of Transition type
        self.buffer_X = np.zeros((buffer_size, self.num_features))
        self.buffer_action = np.zeros((buffer_size))
        self.buffer_nextX = np.zeros((buffer_size, self.num_features))
        self.buffer_reward = np.zeros((buffer_size))
        self.buffer_counter = 0

        self.training_mode = True

        if initial_beta is not None and self.beta_new:
            assert type(initial_beta) is np.ndarray
            assert initial_beta.shape == (self.num_actions, self.num_features)
            self.beta = initial_beta.copy()
            self.beta_new = False


    def saveModel(self):
        """
        This function saves the current model to the file given by path attribute.
        """
        #assert self.training_mode == True

        # so far, we are only reading num_features and num_actions from the file. 
        # the other attributes are written to the file just in case we will need them at some point.
        config = np.array([
            self.num_features,
            self.num_actions,
            self.alpha,
            self.gamma,
            self.buffer_size,
            self.batch_size
        ])

        file = open(self.path, 'wb')
        np.save(file, self.beta)
        np.save(file, config)
        file.close()

        file = open(self.backup_path, 'wb')
        np.save(file, self.beta)
        np.save(file, config)
        file.close()


    def bufferAddTransition(self, transition):
        """
        Add transition to experience buffer.

        :param transition: tuple of (X, action, nextX, reward)
            X: 1D feature vector of state t [np.ndarray]
            action: integer representing the action [int]
            nextX: 1D feature vector of state t+1 [np.ndarray]
            reward: absolute reward gotten after the transition [int]
        """
        #assert self.training_mode == True
        #assert type(transition) is Transition
        #assert type(transition.X) is np.ndarray and transition.X.shape == (self.num_features,)
        #assert type(transition.nextX) is np.ndarray and transition.nextX.shape == (self.num_features,)
        #assert type(transition.action) is int
        #assert type(transition.reward) is int

        self.buffer_X = np.roll(self.buffer_X, 1, axis=0)
        self.buffer_action = np.roll(self.buffer_action, 1, axis=0)
        self.buffer_nextX = np.roll(self.buffer_nextX, 1, axis=0)
        self.buffer_reward = np.roll(self.buffer_reward, 1, axis=0)

        self.buffer_X[0] = transition.X
        self.buffer_action[0] = transition.action
        self.buffer_nextX[0] = transition.nextX
        self.buffer_reward[0] = transition.reward

        if self.buffer_counter < self.buffer_size:
            self.buffer_counter = self.buffer_counter + 1


    def bufferClear(self):
        """
        Clear experience buffer.
        """
        assert self.training_mode == True
        self.buffer_X = np.zeros((self.buffer_size, self.num_features))
        self.buffer_action = np.zeros(self.buffer_size)
        self.buffer_nextX = np.zeros((self.buffer_size, self.num_features))
        self.buffer_reward = np.zeros(self.buffer_size)
        self.buffer_counter = 0


    def gradientUpdate(self):
        """
        Compute new gradients by considering the transitions from the experience buffer.
        Lecture reference: pp. 159-162
        """
        #assert self.training_mode == True

        if self.buffer_counter < 1:
            return

        if self.batch_size > self.buffer_counter:
            batch_size = self.buffer_counter
        else:
            batch_size = self.batch_size

        # generate a batch (= random subset of the experience buffer) for each beta-vector
        selection = np.zeros((self.num_actions, batch_size), dtype=int)
        for i in range(self.num_actions):
            selection[i] = np.random.choice(self.buffer_counter, size=batch_size, replace=False)

        X = self.buffer_X[selection]             # dim: (num_actions x batch_size x num_features)
        nextX = self.buffer_nextX[selection]     # dim: (num_actions x batch_size x num_features)
        reward = self.buffer_reward[selection]   # dim: (num_actions x batch_size x 1)

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
            self.beta[i] = self.beta[i] + (self.alpha / batch_size) * np.sum((X[i].T * (Y[i] - np.matmul(X[i], self.beta[i]))).T, axis=0)


    def nstep_gradientUpdate(self):
        '''
        This function performs the gradient step of the Q-function in n-step TD Q-learning.
        '''
        assert self.training_mode == True
        assert type(self.buffer_size) is int
        assert type(self.n) is int
        assert self.n > 0

        X = self.buffer_X                   # dim: (buffer_size x num_features)
        nextX = self.buffer_nextX               # dim: (buffer_size x num_features)
        reward = self.buffer_reward             # dim: (buffer_size x 1)

        # calculate current guess of Q-function: 
        maxQ = np.max(np.matmul(nextX, self.beta.T),axis=1)

        Y = np.zeros(self.buffer_size)
        # calculate response Y 
        # the reward array will be used to store the response Y
        for i in range(self.buffer_size -self.n): # I think this is expensive as long as not vectorized ...to be continued
            Y[i] = np.dot(np.array(reward[i+1:i+1+self.n])[None,...],np.array([self.gamma**i for i in range(self.n)])[...,None]) + self.gamma**self.n*maxQ[i+self.n]
        for i in range(self.n):
            Y[i+self.n] = np.array(reward)[i+self.n] + (self.gamma * maxQ[i+self.n])
        
        
        # generate a batch (= random subset of the experience buffer) for each beta-vector
        selection = np.zeros((self.num_actions, self.batch_size), dtype=int)
        for i in range(self.num_actions):
            selection[i] = np.random.choice(self.buffer_size, size=self.batch_size, replace=False)

        X = self.buffer_X[selection]             
        nextX = self.buffer_nextX[selection]     
        reward = Y[selection]
        '''
        print("self.beta[0] is")
        print(self.beta[0].shape)
        print(self.beta[0])
        print("self.beta[1] is")
        print(self.beta[1].shape)
        print(self.beta[1])
        '''
        #calculate new betas as in gradientUpdate:
        for i in range(self.num_actions):
            self.beta[i] = self.beta[i] + (self.alpha / self.batch_size) * np.sum((X[i].T * (Y[i] - np.matmul(X[i], self.beta[i]))).T, axis=0)
      

    def Q(self, X, a):
        """
        This is the action value function. It returns a value for a given combination of a 1D feature vector and 
        an action. It can be used to find the best action for the current game state.
        """
        #assert type(X) is np.ndarray and X.shape == (self.num_features,)
        #assert a < self.num_actions and a >= 0
        return np.matmul(X, self.beta[a])


    def predictAction(self, X):
        """
        Based on what the model has learned so far, this function returns the best action for a given 
        1D feature vector representing the game state.
        This function is basically just calculating the return of the Q-function for every possible action.
        """
        #assert type(X) is np.ndarray and X.shape == (self.num_features,)
        return np.argmax(np.matmul(X, self.beta.T))
