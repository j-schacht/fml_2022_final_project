import random
import numpy as np
from .qlearning import *

# this library is needed for the calculation of some features which we are not using
# from igraph import *

# These includes are needed if you want to train the agent using act() functions from other agents
# from agent_code.coin_collector_agent.callbacks import act as coin_collector_act
# from agent_code.rule_based_agent.callbacks import act as rule_based_act

# all possible actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is the epsilon to start the training with. 
# Will be decreased according to EPSILON_DECREASE (see train.py)
EPSILON_START = 0.5

# number of features that we are currently using (= length of feature vector)
NUM_FEATURES = 14

# This can be used to address single features in the feature vector.
# In case of directed features: 
# U = up, R = right, D = down, L = left, M = middle
class Feature:
    COIN_DENSITY_U      = 0
    COIN_DENSITY_R      = 1
    COIN_DENSITY_D      = 2
    COIN_DENSITY_L      = 3
    ESCAPE_U            = 4
    ESCAPE_R            = 5
    ESCAPE_D            = 6
    ESCAPE_L            = 7
    ESCAPE_M            = 8
    CRATE_DENSITY_U     = 9
    CRATE_DENSITY_R     = 10
    CRATE_DENSITY_D     = 11
    CRATE_DENSITY_L     = 12
    CORNERS_AND_BLAST   = 13


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
    self.model = QLearningModel(NUM_FEATURES, len(ACTIONS), logger=self.logger)
    #print(self.model.beta)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # epsilon-greedy policy:
    if self.train and random.random() < self.epsilon:                                  
        self.logger.debug("Choosing action according to epsilon-policy.")
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])      # 80%: walk in any direction. 10% wait. 10% bomb.
        #action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, .0])     # 80%: walk in any direction. 20% wait. 0% bomb.
        #action = coin_collector_act(self, game_state)                      # use act() from coin_collector_agent instead of random.
        #action = rule_based_act(self, game_state)                          # use act() from rule_based_agent instead of random.
    
    else:
        self.logger.debug("Querying model for action.")

        # in learning mode, the feature vector for the current state is already computed in 
        # game_events_occured(), so we don't have to compute it again.
        if not self.train or not hasattr(self, 'current_features'):
            self.current_features = state_to_features(game_state)

        action = ACTIONS[self.model.predictAction(self.current_features)]

    self.logger.debug(f"Chose action {action}")
    #print_features(self.current_features)
    #print(game_state['bombs'])
    #print(action)
    return action


def print_features(feature_vector):
    """
    This function prints the feature vector in a better readable way. Helpful for debugging.
    """
    print(f"COIN_DENSITY     : {feature_vector[Feature.COIN_DENSITY_U:Feature.COIN_DENSITY_L+1]}")
    print(f"ESCAPE           : {feature_vector[Feature.ESCAPE_U:Feature.ESCAPE_M+1]}")
    print(f"CRATE_DENSITY    : {feature_vector[Feature.CRATE_DENSITY_U:Feature.CRATE_DENSITY_L+1]}")
    print(f"CORNERS_AND_BLAST: {feature_vector[Feature.CORNERS_AND_BLAST]}")


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
        bombsmap[bomb[0][0]][bomb[0][1]] = 1

    bombsmapcounter = np.zeros((cols,rows))
    for bomb in bombs:
        bombsmapcounter[bomb[0][0]][bomb[0][1]] = 4-bomb[1]

    wallsmap = (abs(field)-field)*0.5
    notwallsmap = -wallsmap+1
    cratesmap = (field+abs(field))*0.5

    # map of spaces that are free to move on
    freefield = np.ones((cols,rows)) -wallsmap -bombsmap -othersmap -cratesmap -explosionmap
    
    # map of spaces that have blastable objects
    blastablesmap = cratesmap + othersmap*2

    # matrix for the density calculations
    crossmatrix = np.array([[
        1 if abs(i-j)==1 else 0 for i in range(cols)
        ] for j in range(rows)])

    uppermatrix = np.array([[
        1 if i-j ==1 else 0 for i in range(cols)
        ] for j in range(rows)])

    #freedommap = densitymap(freefield, freefield, crossmatrix, weight = 0.1, exponent = 1, iterations = 10)
    #features['freedomdensity'] = neighborvalues(ownpos, freedommap)

    coindensmap = densitymap(coinmap, freefield, crossmatrix, weight = 0.2, exponent = 1, iterations = 12)
    features['coindensity'] = neighborvalues(ownpos, coindensmap*freefield)
    features['coindensity'].pop(4) # the own position does not contain coins
    
    cratedensmap = densitymap(cratesmap, notwallsmap, crossmatrix, weight = 0.3, exponent = 1, iterations = 12)
    features['cratedensity'] = neighborvalues(ownpos, cratedensmap*freefield)
    features['cratedensity'].pop(4) # the own position does not contain crates
    
    dangermap = dangerzones(bombsmapcounter, notwallsmap, crossmatrix, uppermatrix)
    if sum(neighborvalues(ownpos, dangermap + explosionmap)) != 0:
        spacemap = densitymap(freefield, freefield, crossmatrix, weight = 0.5, exponent = 1, iterations = 3)
        escapemap = spacemap - dangermap
        escapemap = escapemap - np.min(escapemap)
        escapemap = escapemap/np.max(escapemap)*(np.ones((cols,rows)) - explosionmap)*freefield
        escapevalues = np.array(neighborvalues(ownpos, escapemap))**3
        escapevalues = escapevalues/np.max(escapevalues)
        features['escape'] = escapevalues.tolist()
    else:
        features['escape'] = [0]*5
    
    # number of free corners in each direction
    features['freecorners'] = find_corners(ownposmap, freefield, crossmatrix, uppermatrix) # TODO: direction wrong

    # number of objects that can be destroyed in each direction
    features['blastables'] = find_blastables(ownposmap, blastablesmap, notwallsmap, crossmatrix, uppermatrix) # TODO: direction wrong

    # feature to determine wether a bomb should be dropped
    if game_state['self'][2]:
        freecorners = sum(features['freecorners'])
        features['cornersandblast'] = [sum(features['blastables'])*freecorners/(freecorners+1)]
    else:
        features['cornersandblast'] = [0.0]

    '''
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
    # size of features_
    #   - coindensity       4
    #   - cratedensity      4
    #   - escape            5
    #   - cornersandblast   1
    #   - bombexplcombined  1
    # freecorners:4
    # blastables:4
    usedfeatures = ['coindensity', 'escape', 'cratedensity', 'cornersandblast']
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
    densmap = densmap/(np.max(densmap)+1)
    densmap = densmap+objectmap
    return densmap


def neighborvalues(position, valuefield):
    # returns the values of the field of the four adjacent tiles 
    # and the current tile as a list
    neighborval = []
    for dirneigh in neighborpositions(position):
        neighborval.append(valuefield[dirneigh[0]][dirneigh[1]])
    return neighborval


def find_corners(ownmap, freemap, crossmatrix, uppermatrix):
    freecorners = []
    for i in range(4):
        numberofcorners = 0
        cornermap = ownmap.copy()
        for j in range(3):
            cornermap = np.matmul(uppermatrix,cornermap)*freemap
            numberofcorners += np.sum(np.matmul(cornermap,crossmatrix)*freemap)
        freecorners.append(numberofcorners)
        ownmap = np.rot90(ownmap)
        freemap = np.rot90(freemap)
    return freecorners


def find_blastables(ownmap, blastablesmap, notwallsmap, crossmatrix, uppermatrix):
    blastables = []
    for i in range(4):
        numberofblastables = 0
        blastmap = ownmap.copy()
        for j in range(3):
            blastmap = np.matmul(uppermatrix,blastmap)*notwallsmap
            numberofblastables += np.sum(blastmap*blastablesmap)
        blastables.append(numberofblastables)
        ownmap = np.rot90(ownmap)
        notwallsmap = np.rot90(notwallsmap)
        blastablesmap = np.rot90(blastablesmap)
    return blastables


def dangerzones(bombsmapcounter, notwallsmap, crossmatrix, uppermatrix):
    dangerzone = np.zeros((bombsmapcounter.shape[0],bombsmapcounter.shape[1]))
    for i in range(4):
        blastmap = bombsmapcounter.copy()
        for j in range(3):
            blastmap = np.matmul(uppermatrix,blastmap)*notwallsmap
            dangerzone = dangerzone + blastmap*(3-j)
        bombsmapcounter = np.rot90(bombsmapcounter)
        notwallsmap = np.rot90(notwallsmap)
        dangerzone = np.rot90(dangerzone)
    dangerzone = dangerzone + bombsmapcounter*4
    return dangerzone


def features_dict_to_array(features, usedfeatures):
    featurearray = []
    for usedfeature in usedfeatures:
        featurearray.extend(features[usedfeature])
    return np.array(featurearray)
