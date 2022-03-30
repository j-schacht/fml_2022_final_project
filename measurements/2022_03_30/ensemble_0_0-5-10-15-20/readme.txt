Hyperparameters:
N     = [0, 5, 10, 15, 20]
ALPHA = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
GAMMA = [0.4, 0.4, 0.4, 0.4, 0.4]

1. Measurement
python main.py play --agents ensemble2 --scenario crate-heaven --train 1 --n-rounds 10000 --no-gui

EPSILON_START    = 1.0
EPSILON_DECREASE = 0.9997
EPSILON_MIN      = 0.2

2. Measurement
python main.py play --agents ensemble2 rule_based_agent rule_based_agent rule_based_agent --scenario classic --train 1 --n-rounds 100000 --no-gui

EPSILON_START    = 1.0
EPSILON_DECREASE = 0.99995
EPSILON_MIN      = 0.05

Stopped at 63631/100000 after 13 hours
