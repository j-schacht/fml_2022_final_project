# fml_2022_final_project

Description of all included agents:


| ensemble             | base version of the ensemble. Not used for measurements.                                                        |
|----------------------|-----------------------------------------------------------------------------------------------------------------|
|                      |                                                                                                                 |
| ensemble_1           | ensemble with N = 1..20 and optimal hyperparameters, mode 1                                                     |
| ensemble_2           | ensemble with N = 1..20 and optimal hyperparameters, mode 0                                                     |
| ensemble_3           | ensemble with N = 1..20 and optimal hyperparameters, mode 2 with decision_prob = 0.1                            |
| ensemble_4           | ensemble with N = [0,0,0,0,1,5,10,20] and optimal hyperparameters, mode 1                                       |
| ensemble_5           | ensemble with N = [0,0,0,0,1,5,10,20] and optimal hyperparameters, mode 0                                       |
| ensemble_6           | ensemble with N = [0,0,0,0,1,5,10,20] and optimal hyperparameters, mode 2 with decision_prob = 0.1              |
| ensemble_7           | ensemble with 8 models, N=0, different hyperparameters, mode 0                                                  |
| ensemble_8           | ensemble with 20 models, N=0, optimal hyperparameters, mode 0                                                   |
| ensemble_9           | ensemble put together from the models trained in solid_snake_h1 - solid_snake_h9, mode 0                        |
|                      |                                                                                                                 |
| solid_snake_original | the original version of solid_snake which has been submitted for tournament. Unfortunately, this had some bugs. |
| solid_snake          | base version of solid_snake (with bugfixes). Trained for 10000 episodes against rule_based_agent in classic.    |
|                      |                                                                                                                 |
| solid_snake_h1       | used for hyperparameter evaluation, alpha=1e-3, gamma=0.6                                                       |
| solid_snake_h2       | used for hyperparameter evaluation, alpha=1e-3, gamma=0.5                                                       |
| solid_snake_h3       | used for hyperparameter evaluation, alpha=1e-3, gamma=0.4                                                       |
| solid_snake_h4       | used for hyperparameter evaluation, alpha=1e-4, gamma=0.6                                                       |
| solid_snake_h5       | used for hyperparameter evaluation, alpha=1e-4, gamma=0.5                                                       |
| solid_snake_h6       | used for hyperparameter evaluation, alpha=1e-4, gamma=0.4                                                       |
| solid_snake_h7       | used for hyperparameter evaluation, alpha=1e-5, gamma=0.6                                                       |
| solid_snake_h8       | used for hyperparameter evaluation, alpha=1e-5, gamma=0.5                                                       |
| solid_snake_h9       | used for hyperparameter evaluation, alpha=1e-5, gamma=0.4                                                       |
|                      |                                                                                                                 |
| solid_snake_q1       | trained with n-step q-learning (N=1), optimal hyperparameters                                                   |
| solid_snake_q2       | trained with n-step q-learning (N=5), optimal hyperparameters                                                   |
| solid_snake_q3       | trained with n-step q-learning (N=10), optimal hyperparameters                                                  |
| solid_snake_q4       | trained with n-step q-learning (N=20), optimal hyperparameters                                                  |
|                      |                                                                                                                 |
|                      |                                                                                                                 |
