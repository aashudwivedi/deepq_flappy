import numpy as np
import skimage

from collections import deque
from keras import layers, models, optimizers
from game import wrapped_flappy_bird as flappy

DO_UP = 0
DO_NOTHING = 1
ACTIONS = [DO_NOTHING, DO_UP]

MAX_REPLAY_SIZE = 50000
OBSERVERATION = 3200
BATCH_SIZE = 100

EPSILON = 1.0
GAMMA = 0.95

epsilon = 1.0


class ReplayMemory(object):
    MAX_SIZE = MAX_REPLAY_SIZE

    def __init__(self):
        self.queue = deque()

    def remember(self, item):
        self.queue.append(item)
        if len(self.queue) > ReplayMemory.MAX_SIZE:
            self.queue.popleft()

    def sample(self, batch_size):
        np.random.choice(self.queue, batch_size, replace=False)


def pre_process_image(image):
    skimage.color.rgb2gray(image)
    image = skimage.transform.resize(image, (80, 80))
    return skimage.exposure.rescale_intensity(image, out_range=(0, 255))


def get_model(input_shape, n_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                            activation='relu', input_shape=input_shape,
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                            activation='relu', input_shape=input_shape,
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                            activation='relu', input_shape=input_shape,
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=512, activation='relu',
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros'))

    model.add(layers.Dense(units=n_classes,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros'))

    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss="mse")
    model.summary()
    return model


def decay_epsilon(epsilon):
    return epsilon


def best_action(model, state):
    q_vals = model.predict(state)
    return ACTIONS[np.argmax(q_vals)]


def act(model, current_state):
    if np.random.random() < EPSILON:
        action = ACTIONS[np.random.random() * 2]
    else:
        action = best_action(model, current_state)
    return action


def replay(model, memory, batch_size):
    minibatch = memory.sample(batch_size)
    for state, action, reward, next_state, is_terminal in minibatch:
        target = reward
        if not is_terminal:
            target = reward + GAMMA * np.max(model.predict(next_state))

        rewards = model.predict(state)
        rewards[action] = target
        model.fit(state, rewards, epochs=1, verbose=0)
        decay_epsilon()


def train():
    game = flappy.GameState()
    game.frame_step()















