from collections import namedtuple, deque
import pickle
from typing import List
import events as e
from .callbacks import state_to_features
from agent_code.ml_agent_1.qlearning import *

# Hyper parameters 
ALPHA =         1     # neither reward nor transition function change over time
GAMMA =         0.9
BUFFER_SIZE =   100
BATCH_SIZE =    40
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

    self.model.setupTraining(ALPHA, GAMMA, BUFFER_SIZE, BATCH_SIZE, autosave=True)

    
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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    t = Transition(
        state_to_features(old_game_state),
        self_action,
        state_to_features(new_game_state),
        reward_from_events(self, events)
    )

    self.model.bufferAddTransition(t)
    self.model.gradientUpdate()


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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    self.model.saveModel()


def reward_from_events(self, events: List[str]) -> int:
    
    game_rewards = {
        e.MOVED_LEFT: -.1,
        e.MOVED_RIGHT: -.1,
        e.MOVED_UP: -.1,
        e.MOVED_DOWN: -.1,
        e.WAITED: -.3,
        e.INVALID_ACTION: -3,

        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: .5,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 1,

        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,
        e.GOT_KILLED: -5,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 3,
        #PLACEHOLDER_EVENT: -.1  
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
