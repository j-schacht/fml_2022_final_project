from collections import namedtuple
import numpy as np
from os.path import exists
import os.path
from datetime import datetime

Transition = namedtuple('Transition', ('X', 'action', 'nextX', 'reward'))

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
        numpy array of booleans with dimension num_features. elements are true if beta has been initialized 
        with normal distribution for the respective feature, i.e. no initial beta was given and no existing 
        model has been loaded from file, or the loaded model did not contain the feature yet.

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

    attribute: logger
        optional. a given logger object can be used to output debugging information
    """

    def __init__(self, num_features, num_actions, path="model", logger=None):
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
        self.logger = logger
        self.path = path + ".npy"

        dir = os.path.dirname(path)
        file = os.path.basename(path)
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.backup_path = os.path.join(dir, "model_backup", file + "_" + date_time + ".npy")

        self.logger.info(f"Initializing Q-Learning Model with {num_features} features and {num_actions} actions.")
        self.logger.info(f"Model path: {self.path}")
        self.logger.info(f"Model backup path: {self.backup_path}")

        self.training_mode = False

        if exists(self.path):
            self.logger.info("Reading existing model from file.")
            file = open(self.path, 'rb')
            self.beta = np.load(file)
            config = np.load(file)
            file.close()

            assert int(config[0]) <= num_features
            assert int(config[1]) == num_actions
            assert self.beta.shape == (int(config[1]), int(config[0]))

            self.beta_new = np.zeros(int(config[0]), dtype=bool)

            if int(config[0]) < num_features:
                self.logger.info(f"The existing model has {int(config[0])} features. Adding {num_features-int(config[0])} more features initialized with uniform distribution.")
                self.beta = np.append(self.beta, np.random.uniform(low=0.0, high=1.0, size=(num_actions, num_features-int(config[0]))), axis=1)
                self.beta_new = np.append(self.beta_new, np.ones(num_features-int(config[0]), dtype=bool))

        else: 
            self.logger.info("No existing model found. Initializing new model with uniform distribution.")
            self.beta = np.random.uniform(low=0.0, high=1.0, size=(num_actions, num_features))
            self.beta_new = np.ones(num_features, dtype=bool)


    def setupTraining(self, alpha, gamma, buffer_size, batch_size, n=0, initial_beta=None):
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

        self.logger.info(f"Setting up Q-Learning Model for training.")
        self.logger.info(f"Hyperparameters: alpha = {alpha}, gamma = {gamma}, buffer_size = {buffer_size}")
        if n > 0:
            self.logger.info(f"n-step Q-Learning will be used instead of normal Q-Learning. N = {n}, NN = {nn}")

        self.alpha = alpha
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n = n
        #self.nn = nn                                                                      # TODO: remove/replace this parameter everywhere
        self.gamma_matrix = np.zeros((buffer_size,buffer_size))                            #temporary ?

        # one buffer for each attribute of Transition type
        self.buffer_X = np.zeros((buffer_size, self.num_features))
        self.buffer_action = np.zeros((buffer_size))
        self.buffer_nextX = np.zeros((buffer_size, self.num_features))
        self.buffer_reward = np.zeros((buffer_size))
        self.buffer_counter = 0

        self.training_mode = True

        if initial_beta is not None:
            assert type(initial_beta) is np.ndarray
            assert initial_beta.shape == (self.num_actions, self.num_features)

            self.logger.info("Initial values for beta are given. All newly initialized features will be set according to these values.")
            self.beta[:,self.beta_new] = initial_beta[:,self.beta_new]
            self.beta_new[:] = False


    def saveModel(self):
        """
        This function saves the current model to the file given by path attribute.
        """
        assert self.training_mode == True

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
        assert self.training_mode == True
        assert type(transition) is Transition
        assert type(transition.X) is np.ndarray and transition.X.shape == (self.num_features,)
        assert type(transition.nextX) is np.ndarray and transition.nextX.shape == (self.num_features,)
        assert type(transition.action) is int
        assert type(transition.reward) is int

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
        assert self.training_mode == True

        if self.buffer_counter < 1:
            return

        X = self.buffer_X                       # dim: (buffer_size x num_features)
        nextX = self.buffer_nextX               # dim: (buffer_size x num_features)
        reward = self.buffer_reward             # dim: (buffer_size x 1)
        action = self.buffer_action

        # find maximum value of Q for nextX and any possible action
        # lecture reference: p. 160 
        # TODO: these dimensions are not right anymore
        # dim:  max((num_actions x num_features) * (num_actions x num_features)^T, axis=2)
        #     = max((num_actions x num_actions), axis=1)
        #     = (num_actions x 1) 
        maxQ = np.max(np.matmul(nextX, self.beta.T), axis=1)

        # calculate expected response Y
        # # lecture reference: p. 160
        # dim:  (num_actions x batch_size x 1) + ((1x1) * (num_actions x batch_size x 1))
        #     = (num_actions x batch_size x 1)
        Y = reward + (self.gamma * maxQ)

        # generate a batch (= random subset of the experience buffer) for each beta-vector
        for i in range(self.num_actions):
            sel = np.where(action == i)[0]

            # calculate the new beta-vectors (= gradient update)
            if sel.size > 0:
                # lecture reference: p. 162
                # dim:  (num_features x 1) + (1x1) * sum(((batch_size x num_features)^T . ((batch_size x 1) - (batch_size x num_features) * (num_features x 1)))^T)
                #     = (num_features x 1) + (1x1) * sum(((batch_size x num_features)^T .  (batch_size x 1))^T)
                #     = (num_features x 1) + (1x1) * sum((batch_size x num_features), axis=0)
                #     = (num_features x 1) + (1x1) * (num_features x 1)
                #     = (num_features x 1)
                self.beta[i] = self.beta[i] + (self.alpha / sel.size) * np.sum((X[sel].T * (Y[sel] - np.matmul(X[sel], self.beta[i]))).T, axis=0)

        #print(self.beta)


    def nstep_gradientUpdate(self):

            #This function performs the gradient step of the Q-function in n-step TD Q-learning.
            
            assert self.training_mode == True
            assert type(self.buffer_size) is int
            assert self.buffer_size > self.n
            assert type(self.n) is int
            assert self.n > 0
            #assert self.nn > 0

            if np.max(self.gamma_matrix) == 0: # create a matrix to multiply reward with in nstep Ql
                gammat = [0] + [self.gamma**(self.n-1-i) for i in range(self.n)]
                for i in range(self.buffer_size-(self.n+1)):
                    gammat = gammat + [0 for i in range(self.buffer_size-self.n+1)] + [self.gamma**(self.n-1-i) for i in range(self.n)]

                gammat = np.asarray(gammat).reshape(self.buffer_size-self.n,self.buffer_size)
                bla = np.concatenate((np.eye(self.n, dtype='float'),np.zeros((self.n,self.buffer_size-self.n))),axis=1)
                self.gamma_matrix = np.concatenate((bla,gammat),axis=0)

            X = self.buffer_nextX                   # dim: (buffer_size x num_features)
            nextX = self.buffer_nextX               # dim: (buffer_size x num_features)
            reward = self.buffer_reward             # dim: (buffer_size x 1)
            action = self.buffer_action

            #calculate maxQ
            maxQ = np.max(np.matmul(nextX, self.beta.T), axis=1)

            # calculate matrix to multiply with current guess of Q-function:
            bla1 = np.zeros((self.n,self.buffer_size-self.n))
            bla2 = self.gamma**self.n * np.eye((self.buffer_size-self.n), dtype='float')
            bla3 = np.zeros((self.buffer_size-self.n,self.n))
            bla4 = self.gamma * np.eye((self.n))
            temp = np.concatenate((np.concatenate((bla4,bla1),axis=1),np.concatenate((bla2,bla3),axis=1)),axis=0)


            Y = np.matmul(self.gamma_matrix,reward) + np.matmul(temp,maxQ)

            # generate the batch of actions for each beta-vector
            sel = np.zeros((self.num_actions), dtype=np.ndarray)
            for i in range(self.num_actions):
                sel[i] = np.where(action == i)[0]

            # calculate the new beta-vectors as in gradientUpdate:
            for i in range(self.num_actions):
                if sel[i].size > 0:
                    self.beta[i] = self.beta[i] + (self.alpha / sel[i].size) * np.sum((X[sel[i]].T * (Y[sel[i]] - np.matmul(X[sel[i]], self.beta[i]))).T, axis=0)


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
