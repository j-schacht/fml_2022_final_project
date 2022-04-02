from typing import List
import events as e
from .callbacks import DECISION_MODE, NUM_FEATURES, state_to_features
from .callbacks import ACTIONS
from .callbacks import Feature as F
from .qlearning import *
from datetime import datetime

# --- HYPERPARAMETERS ---
# EPSILON_START is found in callbacks.py
EPSILON_DECREASE    = 0.99967
EPSILON_MIN         = 0.1
BUFFER_SIZE         = 50

# the N in N-step Q-learning
N     = [0, 0, 0, 0, 0, 0, 0, 0, 0]
ALPHA = [0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.0001, 0.0001, 0.0001]
GAMMA = [0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4]

# Turn output of measurement file on or off
MEASUREMENT = True

# Events
MOVED_TO_COIN = 'MOVED_TO_COIN'
MOVED_FROM_COIN = 'MOVED_FROM_COIN'
MOVED_TO_CRATE = 'MOVED_TO_CRATE'
MOVED_FROM_CRATE = 'MOVED_FROM_CRATE'
MOVED_FROM_BOMBEXPL = 'MOVED_FROM_BOMBEXPL'
MOVED_TO_BOMBEXPL = 'MOVED_TO_BOMBEXPL'
PLACED_BOMB_WELL = 'PLACED_BOMB_WELL'
PLACED_BOMB_VERY_WELL = 'PLACED_BOMB_VERY_WELL'
PLACED_BOMB_EXTREMELY_WELL = 'PLACED_BOMB_EXTREMELY_WELL'
WAITED_TOO_LONG = 'WAITED_TOO_LONG'
RUN_IN_LOOP = 'RUN_IN_LOOP'


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # set hyperparameters
    self.n = N
    self.alpha = ALPHA
    self.gamma = GAMMA
    self.buffer_size = BUFFER_SIZE

    # setup counters
    self.counter = 0
    self.buffer_counter = np.zeros(self.num_models)
    self.counter_rewards = 0
    self.counter_waiting = 0
    self.counter_loop = 0           # used to detect local loops
    self.last_actions = [4,4,4]     # used to detect local loops (initially filled with action 'WAIT')

    for i in range(self.num_models):
        self.models[i].setupTraining(ALPHA[i], GAMMA[i], BUFFER_SIZE, n=N[i])

    # generate file name for measurements and store measurement parameters for documentation
    if MEASUREMENT:
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.measurement_file = f"measurement_{date_time}.csv"

        model_paths = ""
        for i in range(self.num_models-1):
            model_paths += "                   "
            model_paths += self.models[i+1].backup_path
            model_paths += "\n"

        measurement_info = (
            f"FILE             = {self.measurement_file}\n"
            f"NUM_MODELS       = {self.num_models}\n"
            f"DECISION_MODE    = {DECISION_MODE}\n"
            f"MODELS           = {self.models[0].backup_path}\n{model_paths}"
            f"EPSILON_START    = {str(self.epsilon)}\n"
            f"EPSILON_DECREASE = {str(EPSILON_DECREASE)}\n"
            f"EPSILON_MIN      = {str(EPSILON_MIN)}\n"
            f"ALPHA            = {str(ALPHA)}\n"
            f"GAMMA            = {str(GAMMA)}\n"
            f"N                = {str(N)}\n"
            f"BUFFER_SIZE      = {str(BUFFER_SIZE)}\n"
            f"NUM_FEATURES     = {str(NUM_FEATURES)}\n"
            f"COMMAND          = python main.py play [add here]\n\n"
            f"-------------------------------------------------------------------------------\n\n"
        )
        file = open('measurement_info.txt', 'a')
        file.write(measurement_info)
        file.close()


def custom_events(self, old_game_state, self_action, events):
    # calculate feature vector for old_game_state if not already calculated
    if not hasattr(self, 'current_features'):
        self.current_features = state_to_features(old_game_state)

    old_features = self.current_features
    last_action = ACTIONS.index(self_action)

    # count how long the agent is already waiting
    if self_action == 'WAIT':
        self.counter_waiting += 1
    else:
        self.counter_waiting = 0

    # detect local loops
    self.last_actions.append(last_action)
    self.last_actions.pop(0)

    if (self.last_actions[0] == self.last_actions[2] and self.last_actions[0] != self.last_actions[1] 
        and 4 not in self.last_actions and 5 not in self.last_actions):
        self.counter_loop += 1
    else:
        self.counter_loop = 0

    coindensity = old_features[F.COIN_DENSITY_U:F.COIN_DENSITY_L+1]
    escape = old_features[F.ESCAPE_U:F.ESCAPE_M+1]
    cratedensity = old_features[F.CRATE_DENSITY_U:F.CRATE_DENSITY_L+1]
    cornersandblast = old_features[F.CORNERS_AND_BLAST]

    # define custom events 
    if last_action == np.argmax(coindensity) and np.max(coindensity) != 0:
        events.append("MOVED_TO_COIN")

    if last_action != np.argmax(coindensity) and np.max(coindensity) != 0:
        events.append("MOVED_FROM_COIN")

    if last_action == np.argmax(escape) and np.max(escape) != 0: # 0 means no bombs
        events.append("MOVED_FROM_BOMBEXPL")

    if last_action != np.argmax(escape) and np.max(escape) != 0: # 0 means no bombs
        events.append("MOVED_TO_BOMBEXPL")

    if last_action == np.argmax(cratedensity) and np.max(cratedensity) != 0:
        events.append("MOVED_TO_CRATE")

    if last_action != np.argmax(cratedensity) and np.max(cratedensity) != 0:
        events.append("MOVED_FROM_CRATE")

    if self_action == 'BOMB' and cornersandblast >= 1.49:    # at least three blastables and one corner to hide
        events.append("PLACED_BOMB_EXTREMELY_WELL")

    elif self_action == 'BOMB' and cornersandblast >= 0.99:  # at least two blastables and one corner to hide
        events.append("PLACED_BOMB_VERY_WELL")

    elif self_action == 'BOMB' and cornersandblast >= 0.49:  # at least one blastable and one corner to hide
        events.append("PLACED_BOMB_WELL")

    if self.counter_waiting >= 4:
        events.append("WAITED_TOO_LONG")

    if self.counter_loop >= 4:
        events.append("RUN_IN_LOOP")
    
    return events

    
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
    # for the gradient update, we need both old_game_state and new_game_state not to be None
    if not old_game_state or not new_game_state:
        return

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    events = custom_events(self, old_game_state, self_action, events)

    # calculate reward
    reward = reward_from_events(self, events)
    self.counter_rewards = self.counter_rewards + reward

    # state_to_features is defined in callbacks.py
    # The feature vector for the new state is used here for the first time, so we have to compute it first.
    # It can then be used by every other function without having to call state_to_features() again.
    old_features = self.current_features
    self.current_features = state_to_features(new_game_state)

    # push the current transition to the buffer of the q-learning model
    t = Transition(
        old_features,
        ACTIONS.index(self_action),
        self.current_features,
        reward
    )
    
    for i in range(self.num_models):
        self.models[i].bufferAddTransition(t)

        # do gradient update in q-learning model
        if self.models[i].n == 0:         # Q-learning 
            if self.buffer_counter[i] == self.models[i].buffer_size:
                self.models[i].gradientUpdate()
                self.buffer_counter[i] = 0
            else :
                self.buffer_counter[i] += 1
        else:                   # n-step Q-learning
            if self.buffer_counter[i] == (self.models[i].buffer_size - self.models[i].n):
                self.models[i].nstep_gradientUpdate()
                self.buffer_counter[i] = 0
            else :
                self.buffer_counter[i] += 1


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
    # for the gradient update, we need both old_game_state and new_game_state not to be None
    if not last_game_state:
        return

    self.logger.debug(f'Encountered final game event(s) {", ".join(map(repr, events))}')

    # Idea: Add your own events to hand out rewards
    events = custom_events(self, last_game_state, last_action, events)

    # calculate reward
    reward = reward_from_events(self, events)
    self.counter_rewards = self.counter_rewards + reward

    # state_to_features is defined in callbacks.py
    # The feature vector for the new state is zero, since this is the final state
    old_features = self.current_features
    self.current_features = np.zeros(NUM_FEATURES)

    # push the current transition to the buffer of the q-learning model
    t = Transition(
        old_features,
        ACTIONS.index(last_action),
        self.current_features,
        reward
    )
    
    for i in range(self.num_models):
        self.models[i].bufferAddTransition(t)

        # do the last gradient update
        if self.models[i].n == 0 :    # Q-learning
            self.models[i].gradientUpdate() 
            self.buffer_counter[i] = 0
        else :              # n-step Q-learning
            if self.buffer_counter[i] >= self.models[i].buffer_size:
                self.models[i].nstep_gradientUpdate()
                self.buffer_counter[i] = 0
            else :
                self.models[i].gradientUpdate()
                self.buffer_counter[i] = 0

        # store the model
        self.models[i].saveModel()

    # decrease epsilon
    if self.epsilon * EPSILON_DECREASE >= EPSILON_MIN:
        self.epsilon = self.epsilon * EPSILON_DECREASE
    else:
        self.epsilon = 0.0

    # store measurement results
    if MEASUREMENT:
        file = open(self.measurement_file, 'a')
        file.write(f"{str(last_game_state['round'])},{str(last_game_state['self'][1])},{str(last_game_state['step'])},{str(int(e.KILLED_SELF in events))},{str(self.counter_rewards)}\n")
        file.close()

    # reset reward counter
    self.counter_rewards = 0


def reward_from_events(self, events: List[str]) -> int:
    
    game_rewards = {
        e.MOVED_LEFT: -2,
        e.MOVED_RIGHT: -2,
        e.MOVED_UP: -2,
        e.MOVED_DOWN: -2,
        e.WAITED: -2,
        e.INVALID_ACTION: -10,

        e.BOMB_DROPPED: -15,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 10,

        e.KILLED_OPPONENT: 0,
        e.KILLED_SELF: 0,
        e.GOT_KILLED: -30,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0,

        MOVED_TO_COIN: 6,
        MOVED_FROM_COIN: -6,
        MOVED_TO_CRATE: 1,
        MOVED_FROM_CRATE: -1,
        MOVED_FROM_BOMBEXPL: 20,
        MOVED_TO_BOMBEXPL: -20,

        PLACED_BOMB_WELL: 15,
        PLACED_BOMB_VERY_WELL: 17,
        PLACED_BOMB_EXTREMELY_WELL: 20,

        WAITED_TOO_LONG: -3,
        RUN_IN_LOOP: -6
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
