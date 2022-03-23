from typing import List
import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS
from .qlearning import *
from datetime import datetime

# Hyper parameters 
ALPHA =         0.0001
GAMMA =         0.8
BUFFER_SIZE =   50
BATCH_SIZE =    25
# epsilon is found in callbacks.py

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


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

    self.n = 20
    self.counter = 0
    self.counter_nstep = 0

    self.model.setupTraining(ALPHA, GAMMA, BUFFER_SIZE, BATCH_SIZE, n=self.n)

    # file name for measurements
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
    if not old_game_state or not new_game_state:
        return

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    t = Transition(
        state_to_features(old_game_state),
        ACTIONS.index(self_action),
        state_to_features(new_game_state),
        reward_from_events(self, events)
    )
    '''
    self.model.bufferAddTransition(t)

    if self.counter >= BUFFER_SIZE:
        self.model.gradientUpdate()
        self.counter = self.counter + 1
    else:
        self.counter = self.counter + 1
    '''
    # the following if-else statement replaces the above statement in the case of n_step TD Q-learning
    
    if self.counter_nstep % self.n == 1 and self.counter>= BUFFER_SIZE and BUFFER_SIZE <= self.counter_nstep:
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
    self.counter_nstep = 0

    # store measurement results
    file = open(self.measurement_file, 'a')
    file.write(f"{str(last_game_state['round'])},{str(last_game_state['self'][1])},{str(last_game_state['step'])}\n")
    file.close()

    # activate the next lines in case of n-step Q-learning, such that no information gets lost
    
    if self.counter >= BUFFER_SIZE:
        self.model.gradientUpdate()
    


def reward_from_events(self, events: List[str]) -> int:
    
    game_rewards = {
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -3,
        e.INVALID_ACTION: -30,

        e.BOMB_DROPPED: -50,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 10,

        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -50,
        e.GOT_KILLED: -50,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 30,
        #PLACEHOLDER_EVENT: -.1  
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
