import random
import numpy as np
from igraph import * 
from .qlearning import *

EPSILON_START = 0.5

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
NUM_FEATURES = 4

"""
TODO
- change back epsilon-greedy policy
- handle final game states?
"""

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.epsilon = EPSILON_START
    self.model = QLearningModel(NUM_FEATURES, len(ACTIONS))


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # epsilon-greedy policy:
    if self.train and random.random() < 0.1:                                   #temp
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0])  # only temporary
    else:
        self.logger.debug("Querying model for action.")
        action = ACTIONS[self.model.predictAction(state_to_features(game_state))]

    return action


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
    features = {}

    field = game_state['field']
    bombs = game_state['bombs']
    coins = game_state['coins']
    others = game_state['others']
    explosionmap = game_state['explosion_map']
    ownposx, ownposy = game_state['self'][3]
    ownpos = np.array([ownposx, ownposy])
    cols = field.shape[0] # x
    rows = field.shape[1] # y

    # map with all zeros but the own position
    ownposmap = np.zeros((cols,rows))
    ownposmap[ownposx][ownposy] = 1

    othersmap = np.zeros((cols,rows))
    for other in others:
        othersmap[other[3][0]][other[3][1]] = 1

    coinmap = np.zeros((cols,rows))
    for coin in coins:
        coinmap[coin[0]][coin[1]] = 1

    bombsmap = np.zeros((cols,rows))
    for bomb in bombs:
        bombsmap[bomb[0][0]][bomb[0][1]] = 4-bomb[1]

    wallsmap = (abs(field)-field)*0.5
    notwallsmap = -wallsmap+1
    cratesmap = (field-abs(field))*0.5

    # map of spaces that are free to move on
    freefield = np.ones((cols,rows)) -wallsmap -bombsmap -othersmap
    
    # map of spaces that have blastable objects
    blastablesmap = cratesmap + othersmap

    # matrix for the density calculations
    crossmatrix = np.array([[
        1 if abs(i-j)==1 else 0 for i in range(cols)
        ] for j in range(rows)])

    #freedommap = densitymap(freefield, freefield, crossmatrix, weight = 0.1, exponent = 1, iterations = 10)
    #features['freedomdensity'] = neighborvalues(ownpos, freedommap)

    coindensmap = densitymap(coinmap, freefield, crossmatrix, weight = 0.1, exponent = 1, iterations = 5)
    features['coindensity'] = neighborvalues(ownpos, coindensmap)
    features['coindensity'].pop(4) # the own position does not contain coins

    '''
    bombdensmap = densitymap(bombsmap, notwallsmap, crossmatrix, weight = 0.5, exponent = 1, iterations = 5)
    bombdensmap = -bombdensmap + (freefield-1)*np.sum(bombdensmap)
    features['bombdensity'] = neighborvalues(ownpos, bombdensmap)

    explosiondensmap = densitymap(explosionmap, freefield, crossmatrix, weight = 0.2, exponent = 1, iterations = 1)
    explosiondensmap = -explosiondensmap + (freefield-1)*np.sum(explosiondensmap)
    features['explosiondensity'] = neighborvalues(ownpos, explosiondensmap)
    
    # number of free corners in each direction
    features['freecorners'] = find_corners(ownposmap, freefield, crossmatrix)

    # number of objects that can be destroyed in each direction
    features['blastables'] = find_blastables(ownposmap, blastablesmap, notwallsmap, crossmatrix)

    # calculate distance to the closest coin using graph algorithms
    if len(coins) > 0:
        cols = field.shape[0] # x
        rows = field.shape[1] # y
        coins_np = np.array(coins)
        coins_flat = coins_np[:,0] * cols + coins_np[:,1]

        g = Graph()
        g.add_vertices(cols*rows)

        for x in range(1, cols-2):
            for y in range(1, rows-2):
                if field[x,y] == 0:
                    if field[x+1,y] == 0:
                        g.add_edges([(x*cols+y, (x+1)*cols+y)])
                    if field[x,y+1] == 0:
                        g.add_edges([(x*cols+y, x*cols+y+1)])

        coin_distances = g.shortest_paths(source=ownposx*cols+ownposy, target=coins_flat)
        closest_coin_distance = np.min(coin_distances[0])
        max3 = min(3, len(coin_distances[0]))
        closest_3_coins_distance = np.sum(np.partition(coin_distances[0], max3-1)[0:max3])

        if closest_coin_distance == float("inf"):
            closest_coin_distance = 1000
        if closest_3_coins_distance == float("inf"):
            closest_3_coins_distance = 1000

        features['closest_coin_distance'] = [closest_coin_distance]
        features['closest_3_coins_distance'] = [closest_3_coins_distance]
    else: 
        features['closest_coin_distance'] = [1000]
        features['closest_3_coins_distance'] = [1000]
    
    # check which directions are free to move
    features['up_free'] = [0]
    features['down_free'] = [0]
    features['left_free'] = [0]
    features['right_free'] = [0]

    if field[ownposx,ownposy-1] == 0:
        features['up_free'] = [1]

    if field[ownposx,ownposy+1] == 0:
        features['down_free'] = [1]

    if field[ownposx-1,ownposy] == 0:
        features['left_free'] = [1]

    if field[ownposx+1,ownposy] == 0:
        features['right_free'] = [1]
    '''
    # freedomdensity:5
    # coindensity:4
    # bombdensity:5
    # explosiondensity:5
    # freecorners:4
    # blastables:4
    # closest_coin_distance:1
    # closest_3_coins_distance:1
    #usedfeatures = ['freedomdensity','coindensity','bombdensity','explosiondensity','freecorners','blastables','closest_coin_distance','closest_3_coins_distance']
    usedfeatures = ['coindensity']
    featurearray = features_dict_to_array(features, usedfeatures)
    return featurearray

    
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
        #densmap = densmap/(sum(sum(densmap)))
    densmap = densmap+objectmap
    densmap = densmap/(np.max(densmap)+1)
    return densmap


def neighborvalues(position, valuefield):
    # returns the values of the field of the four adjacent tiles 
    # and the current tile as a list
    neighborval = []
    for dirneigh in neighborpositions(position):
        neighborval.append(valuefield[dirneigh[0]][dirneigh[1]])
    return neighborval


def find_corners(ownmap, freemap, crossmatrix):
    uppermatrix = np.array([[
        1 if i-j ==1 else 0 for i in range(freemap.shape[0])
        ] for j in range(freemap.shape[1])])

    freecorners = []
    for i in range(4):
        numberofcorners = 0
        cornermap = ownmap.copy()
        for j in range(3):
            cornermap = np.matmul(uppermatrix,cornermap)*freemap
            numberofcorners += np.sum(np.matmul(cornermap,crossmatrix)*freemap)
        freecorners.append(numberofcorners/4)
        ownmap = np.rot90(ownmap)
        freemap = np.rot90(freemap)
    return freecorners


def find_blastables(ownmap, blastablesmap, notwallsmap, crossmatrix):
    uppermatrix = np.array([[
        1 if i-j ==1 else 0 for i in range(ownmap.shape[0])
        ] for j in range(ownmap.shape[1])])

    blastables = []
    for i in range(4):
        numberofblastables = 0
        blastmap = ownmap.copy()
        for j in range(3):
            blastmap = np.matmul(uppermatrix,blastmap)*notwallsmap
            numberofblastables += np.sum(blastmap*notwallsmap)
        blastables.append(numberofblastables/3)
        ownmap = np.rot90(ownmap)
        notwallsmap = np.rot90(notwallsmap)
        blastablesmap = np.rot90(blastablesmap)
    return blastables


def features_dict_to_array(features, usedfeatures):
    featurearray = []
    for usedfeature in usedfeatures:
        featurearray.extend(features[usedfeature])
    return np.array(featurearray)
