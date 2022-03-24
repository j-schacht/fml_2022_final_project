from typing import List
import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS
from .qlearning import *
from datetime import datetime


# --- HYPERPARAMETERS ---
# EPSILON_START is found in callbacks.py
EPSILON_DECREASE    = 0.999
EPSILON_MIN         = 0.1
ALPHA               = 0.0001
GAMMA               = 0.6
BUFFER_SIZE         = 50
BATCH_SIZE          = 25

# step size for n-step q-learning (set to zero to use normal q-learning)
N                   = 0

INITIAL_BETA = np.array([[1,-0.1,-0.1,-0.1],
                        [-0.1, 1,-0.1,-0.1],
                        [-0.1,-0.1,1,-0.1],
                        [-0.1,-0.1,-0.1,1],
                        [-0.1,-0.1,-0.1,-0.1],
                        [-0.5,-0.5,-0.5,-0.5],
])

# Measurements
MEASUREMENT =   True

# Events
MOVED_TO_COIN = 'MOVED_TO_COIN'
MOVED_FROM_BOMB = 'MOVED_FROM_BOMB'
MOVED_FROM_EXPLOSION = 'MOVED_FROM_EXPLOSION'

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.alpha = ALPHA
    self.gamma = GAMMA
    self.buffer_size = BUFFER_SIZE
    self.batch_size = BATCH_SIZE

    self.n = N
    self.counter = 0
    self.counter_nstep = 0

    #self.model.setupTraining(ALPHA, GAMMA, BUFFER_SIZE, BATCH_SIZE, n=self.n, initial_beta=INITIAL_BETA)
    self.model.setupTraining(ALPHA, GAMMA, BUFFER_SIZE, BATCH_SIZE, n=self.n)

    # file name for measurements 
    if MEASUREMENT: # TODO: epsilon decreasing!
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.measurement_file = f"measurement_{date_time}_{str(self.epsilon)}_{str(ALPHA)}_{str(GAMMA)}_{str(BATCH_SIZE)}_{str(BUFFER_SIZE)}.csv"

    
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
    if not old_game_state or not new_game_state:    #TODO
        return

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    old_features = state_to_features(old_game_state)
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    featurecounter = 0
    coindensity = old_features[featurecounter:featurecounter+4]
    featurecounter += 4
    #bombdensity = old_features[featurecounter:featurecounter+5]
    #featurecounter += 5
    #explosiondensity = old_features[featurecounter:featurecounter+5]
    #featurecounter += 5
    if ACTIONS.index(self_action) == np.argmax(coindensity) and np.argmax(coindensity) != 0:
        events.append("MOVED_TO_COIN")
    #if ACTIONS.index(self_action) == np.argmax(bombdensity) and np.argmax(bombdensity) != 0:
    #    events.append("MOVED_FROM_BOMB")
    #if ACTIONS.index(self_action) == np.argmax(explosiondensity) and np.argmax(explosiondensity) != 0:
    #    events.append("MOVED_FROM_EXPLOSION")
    
    # state_to_features is defined in callbacks.py
    t = Transition(
        old_features,
        ACTIONS.index(self_action),
        state_to_features(new_game_state),
        reward_from_events(self, events)
    )
    
    self.model.bufferAddTransition(t)

    if self.n == 0:
        # normal q-learning
        self.model.gradientUpdate()
    else:
        # n-step q-learning
        if self.counter_nstep % self.n == 1 and self.counter >= BUFFER_SIZE and BUFFER_SIZE <= self.counter_nstep:
            self.model.nstep_gradientUpdate()
            self.counter = self.counter + 1
            self.counter_nstep = self.counter_nstep + 1
        else:
            self.counter = self.counter + 1
            self.counter_nstep = self.counter_nstep + 1
    

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

    if self.n != 0:
        # only if n-step q-learning
        self.counter_nstep = 0
        # make sure that no information gets lost
        if self.counter >= BUFFER_SIZE:
            self.model.gradientUpdate()
    


def reward_from_events(self, events: List[str]) -> int:
    
    game_rewards = {
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -3,
        e.INVALID_ACTION: -5,

        e.BOMB_DROPPED: -50,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 20,

        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -50,
        e.GOT_KILLED: -50,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 30,

        MOVED_TO_COIN: 1,
        MOVED_FROM_BOMB: 3,
        MOVED_FROM_EXPLOSION: 3,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
