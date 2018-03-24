import numpy as np
import random

from collections import deque
from keras import layers, models, optimizers
from game import wrapped_flappy_bird as flappy

from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer


class GameEnv(object):
    def __init__(self):
        self.game = flappy.GameState()
        image, _, _ = self.game.frame_step(0)
        image = self.pre_process_image(image)
        self.state = np.concatenate([image] * 4, axis=3)

    @staticmethod
    def pre_process_image(image):
        image = color.rgb2gray(image)
        image = transform.resize(image, (80, 80))
        image = exposure.rescale_intensity(image, out_range=(0, 255))
        return image.reshape(1, 80, 80, 1)

    def get_state(self):
        return self.state

    def step(self, action):
        image, reward, done = self.game.frame_step(action)
        image = self.pre_process_image(image)
        self.state[:, :, :, :3] = image
        return self.state, reward, done


class DQNAgent(object):
    ACTIONS = [0, 1]

    def __init__(self, action_size):
        self.action_size = action_size

        self.memory = deque(maxlen=50000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model(self.action_size)

    def _build_model(self, n_classes):
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                                activation='relu', input_shape=(80, 80, 4),
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros'))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                activation='relu',
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros'))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                activation='relu',
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def best_action(self, state):
        q_vals = self.model.predict(state)
        return self.ACTIONS[np.argmax(q_vals)]

    def act(self, state):
        """
        :param state: current state of the game on which act has to be performed
        :return: best action to be performed at this state
        """
        if np.random.random() < self.epsilon:
            action = self.ACTIONS[int(np.random.random() * 2)]
        else:
            action = self.best_action(state)
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def build_replay(game_env, agent):
    state = game_env.get_state()
    for i in range(2000):
        action = np.random.choice([0, 1], p=[0.9, 0.1])
        next_state, reward, done = game_env.step(action)
        agent.remember(state, action, reward, next_state, done)
        if done:
            print(reward)
        state = next_state


def train(episode_count):
    # initialize gym environment and the agent
    agent = DQNAgent(2)
    game_env = GameEnv()
    build_replay(game_env, agent)

    state = game_env.state
    for e in range(episode_count):
        for time_t in range(500):
            action = agent.act(state)
            next_state, reward, done = game_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/{}, score: {}".format(
                    e, episode_count, reward))
                break
        # train the agent with the experience of the episode
        agent.replay(32)


if __name__ == '__main__':
    train(1000)
