import random
from game import wrapped_flappy_bird as flappy

DO_UP = 0
DO_NOTHING = 1

ACTIONS = [DO_NOTHING, DO_UP]

game_state = flappy.GameState()

terminal = False
while not terminal:
    action = ACTIONS[int(random.random() * 2)]
    image_data, reward, terminal = game_state.frame_step(action)
    print(reward)