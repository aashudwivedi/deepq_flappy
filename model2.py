import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
import random

from collections import deque
from keras import layers, models, optimizers, callbacks

from skimage import transform, color, exposure

out_dir = 'output/' if os.path.exists('output/') else '/output/'

RUN_NAME = 'first'

from ple.games.flappybird import FlappyBird
from ple import PLE


class GameEnv(object):
    def __init__(self):
        game = FlappyBird()
        self.p = PLE(game, fps=30, display_screen=False)
        self.p.init()

        image = self.p.getScreenRGB()
        image = self.pre_process_image(image)
        self.state = np.concatenate([image] * 4, axis=3)
        self.score = 0

    @staticmethod
    def pre_process_image(image):
        image = color.rgb2gray(image)
        image = transform.resize(image, (80, 80))
        image = exposure.rescale_intensity(image, out_range=(0, 255))
        return image.reshape(1, 80, 80, 1)

    def get_state(self):
        return self.state

    def step(self, action):
        _ = self.p.act(action)
        image = self.p.getScreenRGB()

        done = False
        if self.p.game_over():
            done = True
            self.p.reset_game()
            reward = -10

        else:
            reward = 0.1

        image = self.pre_process_image(image)
        self.state[:, :, :, :3] = image

        return_score = self.score + reward
        self.score = 0 if done else self.score + reward

        return self.state, reward, done, return_score

    def get_score(self):
        return self.score


class DQNAgent(object):
    ACTIONS = [0, 1]
    MAX_MEMORY = 50000

    def __init__(self, action_size):
        self.action_size = action_size

        self.memory = deque(maxlen=self.MAX_MEMORY)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4
        self.model = self._build_model(self.action_size)
        self.load_weights()
        self.create_data_dir()
        self.callback = callbacks.TensorBoard(
            log_dir=self.data_dir_path(), histogram_freq=0,
            write_graph=True, write_grads=True,
            write_images=True)

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

        optimizer = optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        model.summary()
        return model

    def data_dir_path(self):
        return os.path.join(out_dir, RUN_NAME)

    def _weights_path(self):
        return os.path.join(self.data_dir_path(), '{}.h5'.format('model'))

    def create_data_dir(self):
        if not os.path.exists(self.data_dir_path()):
            os.mkdir(self.data_dir_path())

    def load_weights(self):
        if os.path.exists(self._weights_path()):
            self.model.load_weights(self._weights_path())
        print('loaded weights')

    def save_weights(self):
        model_json = self.model.to_json()
        with open(self._weights_path(), 'w') as f:
            f.write(model_json)
        self.model.save_weights(self._weights_path())

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

    def decrease_epsilon(self, episode):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            print('    q_values : {}'.format(target_f))
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0,
                           callbacks=[self.callback])


def build_replay(game_env, agent):
    state = game_env.get_state()
    for i in range(4000):
        # action = np.random.choice([0, 1], p=[0.9, 0.1])
        action = agent.act(state)
        next_state, reward, done, score = game_env.step(action)
        print(reward)
        agent.remember(state, action, reward, next_state, done)
        if done:
            print('score = {}'.format(score))
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
            next_state, reward, done, score = game_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print('-' * 50)
                print("episode: {}/{}, score: {} epsilonn: {}".format(
                    e, episode_count, score, agent.epsilon))
                print('-' * 50)
                break
        # train the agent with the experience of the episode
        agent.replay(32)
        agent.save_weights()
        agent.decrease_epsilon(e)


if __name__ == '__main__':
    train(1000)

