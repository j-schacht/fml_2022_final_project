from typing import List
import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS
from .qlearning import *
from datetime import datetime

# --- HYPERPARAMETERS ---
# EPSILON_START is found in callbacks.py
EPSILON_DECREASE    = 0.9995
EPSILON_MIN         = 0.1
ALPHA               = 0.0001
GAMMA               = 0.6
BUFFER_SIZE         = 50

# the N in N-step Q-learning
N                   = 0


# this array can be filled with a initial guess for beta, such that the model converges faster
INITIAL_BETA = np.array(
    [[1,-0.1,-0.1,-0.1],
    [-0.1, 1,-0.1,-0.1],
    [-0.1,-0.1,1,-0.1],
    [-0.1,-0.1,-0.1,1],
    [-0.1,-0.1,-0.1,-0.1],
    [-0.5,-0.5,-0.5,-0.5],
])

# Turn output of measurement file on or off
MEASUREMENT = True

# Events
MOVED_TO_COIN = 'MOVED_TO_COIN'
MOVED_TO_CRATE = 'MOVED_TO_CRATE'
MOVED_FROM_BOMBEXPL = 'MOVED_FROM_BOMBEXPL'
PLACED_BOMB_WELL = 'PLACED_BOMB_WELL'

# This can be used to address single features in the feature vector.
# In case of directed features: 
# U = up, R = right, D = down, L = left, M = middle
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


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # set hyperparameters
    self.alpha = ALPHA
    self.gamma = GAMMA
    self.buffer_size = BUFFER_SIZE


    self.n = N
    #self.nn = NN
    self.buffer_counter = 0                                         #changed
    #self.counter_nstep = 0                                         #to be removed

    #self.model.setupTraining(ALPHA, GAMMA, BUFFER_SIZE, n=self.n, initial_beta=INITIAL_BETA)
    self.model.setupTraining(ALPHA, GAMMA, BUFFER_SIZE, n=self.n)


    # file name for measurements 
    if MEASUREMENT:
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.measurement_file = f"measurement_{date_time}_{str(self.epsilon)}_{str(EPSILON_DECREASE)}_{str(EPSILON_MIN)}_{str(ALPHA)}_{str(GAMMA)}_{str(BUFFER_SIZE)}_{str(N)}_{str(self.model.num_features)}.csv"

    
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if not old_game_state or not new_game_state:
        return

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if not hasattr(self, 'current_features'):
        self.current_features = state_to_features(old_game_state)

    old_features = self.current_features
    last_action = ACTIONS.index(self_action)

    coindensity = old_features[COIN_DENSITY_U:COIN_DENSITY_L]
    escape = old_features[ESCAPE_U:ESCAPE_M]
    cratedensity = old_features[CRATE_DENSITY_U:CRATE_DENSITY_L]
    cornersandblast = old_features[CORNERS_AND_BLAST]
    
    if last_action == np.argmax(coindensity) and np.argmax(coindensity) != 0:
        events.append("MOVED_TO_COIN")

    if last_action == np.argmax(escape) and np.argmax(escape) != 0: # 0 means no bombs
        events.append("MOVED_FROM_BOMBEXPL")

    if last_action == np.argmax(cratedensity) and np.argmax(cratedensity) != 0:
        events.append("MOVED_TO_CRATE")

    if self_action == 'BOMB' and cornersandblast >= 1.0:
        events.append("PLACED_BOMB_WELL")

    # state_to_features is defined in callbacks.py
    # The feature vector for the new state is used here for the first time, so we have to compute it first.
    # It can then be used by every other function without having to call state_to_features() again.
    self.current_features = state_to_features(new_game_state)

    t = Transition(
        old_features,
        ACTIONS.index(self_action),
        self.current_features,
        reward_from_events(self, events)
    )
    
    self.model.bufferAddTransition(t)

    if self.n == 0:         # Q-learning 
        if self.buffer_counter == self.buffer_size:
            self.model.gradientUpdate()
            self.buffer_counter = 0
        else :
            self.buffer_counter += 1
    else:                   # n-step Q-learning
        if self.buffer_counter == (self.buffer_size - self.n):
            self.model.nstep_gradientUpdate()
            self.buffer_counter = 0
        else :
            self.buffer_counter += 1


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # store the model
    self.model.saveModel()

    # decrease epsilon
    if self.epsilon * EPSILON_DECREASE >= EPSILON_MIN:
        self.epsilon = self.epsilon * EPSILON_DECREASE

    # store measurement results
    if MEASUREMENT:
        file = open(self.measurement_file, 'a')
        file.write(f"{str(last_game_state['round'])},{str(last_game_state['self'][1])},{str(last_game_state['step'])}\n")
        file.close()

    if self.n == 0 :    # Q-learning
        self.model.gradientUpdate() 
        self.buffer_counter = 0
    else :              # n-step Q-learning
        if self.model.buffer_counter >= self.buffer_size:
            self.model.nstep_gradientUpdate()
            self.buffer_counter = 0
        else :
            self.model.gradientUpdate()
            self.buffer_counter = 0


def reward_from_events(self, events: List[str]) -> int:
    
    game_rewards = {
        e.MOVED_LEFT: -2,
        e.MOVED_RIGHT: -2,
        e.MOVED_UP: -2,
        e.MOVED_DOWN: -2,
        e.WAITED: -4,
        e.INVALID_ACTION: -10,

        e.BOMB_DROPPED: -15,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 10,

        e.KILLED_OPPONENT: 100,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -100,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 100,

        MOVED_TO_COIN: 4,
        MOVED_TO_CRATE: 1,
        MOVED_FROM_BOMBEXPL: 7,
        PLACED_BOMB_WELL: 16
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
