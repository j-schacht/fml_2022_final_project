import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    ### !!code to test features - not the actual model!!
    statefeatures = state_to_features(game_state)
    neighborvalues = 0.001*statefeatures['freedomdensity'] + 1*(statefeatures['coindensity']) + 100*(statefeatures['bombdensity']) + 100*statefeatures['explosiondensity']
    neighborvalueslist = neighborvalues.tolist()
    action = neighborvalueslist.index(max(neighborvalueslist))
    return ACTIONS[action]
    ###

    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    features = {

    }
    field = game_state['field']
    bombs = game_state['bombs']
    coins = game_state['coins']
    others = game_state['others']
    explosionmap = game_state['explosion_map']
    ownposx, ownposy = game_state['self'][3]
    ownpos = np.array([ownposx, ownposy])
    fieldsize = field.shape[0]


    notwalls = (field-abs(field))*0.5+1 # map of spaces that are not walls

    freefield = abs(abs(field)-1) # map of spaces that are free to move on
    for other in others:
        freefield[other[3][0]][other[3][1]] = 0
    for bomb in bombs:
        freefield[bomb[0][0]][bomb[0][1]] = 0
    
    # matrix for the density calculations
    crossmatrix = np.array([
        [
        1 if abs(i-j)==1 else 0 for i in range(fieldsize)
        ] for j in range(fieldsize)
        ])

    freedommap = densitymap(freefield, freefield, crossmatrix, weight = 1, exponent = 1.2, iterations = 10)
    features['freedomdensity'] = neighborvalues(ownpos, freedommap)

    if len(coins) != 0:
        coinmap = np.array([[0]*fieldsize]*fieldsize)
        for coin in coins:
            coinmap[coin[0]][coin[1]] = 1
        coindensmap = densitymap(coinmap, freefield, crossmatrix, weight = 0.5, exponent = 1.3, iterations = 15)
        features['coindensity'] = neighborvalues(ownpos, coindensmap)
        features['coindensity'][4] = 0 # the own position does not contain coins
    else:
        features['coindensity'] = np.array([0]*5)

    if len(bombs) != 0:
        bombmap = np.array([[0]*fieldsize]*fieldsize)
        for bomb in bombs:
            bombmap[bomb[0][0]][bomb[0][1]] = 4-bomb[1]
        bombdensmap = densitymap(bombmap, notwalls, crossmatrix, weight = 0.5, exponent = 1.3, iterations = 5)
        bombdensmap = -bombdensmap + freefield
        features['bombdensity'] = neighborvalues(ownpos, bombdensmap)
    else:
        features['bombdensity'] = np.array([0]*5)

    if sum(sum(explosionmap)) != 0:
        explosiondensmap = densitymap(explosionmap, freefield, crossmatrix, weight = 0.5, exponent = 1.1, iterations = 1)
        explosiondensmap = -explosiondensmap + freefield
        features['explosiondensity'] = neighborvalues(ownpos, explosiondensmap)
    else:
        features['explosiondensity'] = np.array([0]*5)


    return features

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)

    
def neighborpositions(position):
    # returns the four adjacent tiles and the own position as a list
    posx = position[0]
    posy = position[1]
    neighborpos = [[posx,posy-1],[posx+1,posy],[posx,posy+1],[posx-1,posy],[posx,posy]]
    return neighborpos

def densitymap(objectmap, freemap, crossmatrix, weight = 1, exponent = 1, iterations = 10):
    # returns the density map of the objects in the objectmap
    densmap = objectmap.copy()
    for i in range(iterations):
        densmap = np.matmul(densmap,crossmatrix)*weight+np.matmul(crossmatrix,densmap)*weight+densmap
        densmap = (densmap*freemap)**exponent
        densmap = densmap/(sum(sum(densmap)))
    densmap = densmap+objectmap
    densmap = densmap/(sum(sum(densmap)))
    return densmap

def neighborvalues(position, valuefield):
    # returns the values of the field of the four adjacent tiles 
    # and the current tile as a list
    neighborval = []
    for dirneigh in neighborpositions(position):
        neighborval.append(valuefield[dirneigh[0]][dirneigh[1]])
    return np.array(neighborval)
