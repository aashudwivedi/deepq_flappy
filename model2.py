import argparse
import os
import numpy as np
import pickle
import random

from collections import deque
from keras import layers, models, optimizers, callbacks, backend as K
from skimage import transform, color, exposure
from PIL import Image


out_dir = 'output/' if os.path.exists('output/') else '/output/'
in_dir = 'input/' if os.path.exists('input/') else '/input/'

if os.path.exists('input/'):
    is_local = True

GOOD_SCORE = 5.5

RUN_NAME = 'first'

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80


# class QValLogger(TensorBoard):
#     def __init__(self, log_dir, **kwargs):
#         # Make the original `TensorBoard` log to a subdirectory 'training'
#         training_log_dir = os.path.join(log_dir, 'training')
#         super(QValLogger, self).__init__(training_log_dir, **kwargs)
#
#         # Log the validation metrics to a separate subdirectory
#         self.val_log_dir = os.path.join(log_dir, 'validation')
#
#     def set_model(self, model):
#         # Setup writer for validation metrics
#         self.val_writer = tf.summary.FileWriter(self.val_log_dir)
#         super(QValLogger, self).set_model(model)
#
#     def on_epoch_end(self, epoch, logs=None):
#         # Pop the validation logs and handle them separately with
#         # `self.val_writer`. Also rename the keys so that they can
#         # be plotted on the same figure with the training metrics
#         logs = logs or {}
#         val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
#         for name, value in val_logs.items():
#             summary = tf.Summary()
#             summary_value = summary.value.add()
#             summary_value.simple_value = value.item()
#             summary_value.tag = name
#             self.val_writer.add_summary(summary, epoch)
#         self.val_writer.flush()
#
#         # Pass the remaining logs to `TensorBoard.on_epoch_end`
#         logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
#         super(QValLogger, self).on_epoch_end(epoch, logs)
#
#     def on_train_end(self, logs=None):
#         super(QValLogger, self).on_train_end(logs)
#         self.val_writer.close()


def save_state(count, state):
    path = out_dir + RUN_NAME + "/images"
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)

    dir_path = path + "/" + str(count)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # for i in range(4):
    #     array = state[:, :, :, i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    #     array = array * 255
    #     image = Image.fromarray(array, mode='RGB')
    #     image.save(dir_path + "/" + str(i) + ".bmp")

    image = Image.fromarray(state, mode='RGB')
    image.save(dir_path + "/" + 'image' + ".bmp")


class GameEnv(object):

    def __init__(self, display_screen):
        self.width = IMAGE_WIDTH
        self.height = IMAGE_HEIGHT

        self.count = 0
        self.p = PLE(FlappyBird(), fps=30, display_screen=display_screen)
        self.p.init()
        self._update_state()
        self.score = 0

    def pre_process_image(self, image):
        self.count += 1
        image = color.rgb2gray(image)
        image = transform.resize(image, (self.width, self.height))
        image = exposure.rescale_intensity(image, out_range=(0, 255))
        image = image.astype('float')
        image = image / 255.0
        return image.reshape(1, self.width, self.height, 1)

    def _update_state(self):
        image = self.p.getScreenRGB()
        # TODO: convert to float
        image = self.pre_process_image(image)
        state = getattr(self, 'state', None)
        if state is None:
            self.state = np.concatenate([image] * 4, axis=3)
        else:
            self.state[:, :, :, :3] = image

    def get_state(self):
        return self.state

    def step(self, action):
        if action == 1:
            _ = self.p.act(119)
        else:
            _ = self.p.act(None)

        self._update_state()

        done = False
        if self.p.game_over():
            done = True
            self.p.reset_game()
            reward = -1
        else:
            reward = 0.1

        return_score = self.score + reward
        self.score = 0 if done else self.score + reward

        return self.state, reward, done, return_score

    def get_score(self):
        return self.score


class DQNAgent(object):
    ACTIONS = [0, 1]
    MAX_MEMORY = 1000000

    def __init__(self, action_size):
        self.action_size = action_size

        self.memory = deque(maxlen=self.MAX_MEMORY)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.1 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 1e-5
        self.model = self._build_model(self.action_size)
        self.load_weights()
        self.create_data_dir()
        self.callback = callbacks.TensorBoard(
            log_dir=self.data_dir_path(), histogram_freq=0,
            write_graph=True, write_grads=True,
            write_images=True)
        self.count = 0

    def _huber_loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def _build_model(self, n_classes):
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(8, 8), padding='same',
                                activation='relu',
                                input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 4),
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=64, kernel_size=(4, 4), padding='same',
                                activation='relu',
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                activation='relu',
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                activation='relu',
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                activation='relu',
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                activation='relu',
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Flatten())

        model.add(layers.Dense(units=512, activation='relu',
                               kernel_initializer='glorot_normal',
                               bias_initializer='zeros'))

        model.add(layers.Dense(units=512, activation='relu',
                               kernel_initializer='glorot_normal',
                               bias_initializer='zeros'))

        model.add(layers.Dense(units=512, activation='relu',
                               kernel_initializer='glorot_normal',
                               bias_initializer='zeros'))

        model.add(layers.Dense(units=n_classes,
                               kernel_initializer='glorot_normal',
                               bias_initializer='zeros'))

        optimizer = optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss=self._huber_loss)
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

        self.count += 1
        # self.save_state(self.count, state)
        return action

    def decrease_epsilon(self, episode):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        input_shape = [batch_size]
        input_shape.extend(minibatch[0][0].shape[1:])

        x = np.zeros(input_shape)
        y = np.zeros((batch_size, 2))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            y[i] = target_f
            x[i, :, :, :] = state

        self.model.fit(x, y, epochs=1, verbose=0,
                       callbacks=[self.callback])


def load_queue():
    path = os.path.join(in_dir, 'queue.pickle')
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_queue(queue):
    path = os.path.join(in_dir, 'queue.pickle')
    with open(path, 'wb') as f:
        print('saved queue')
        return pickle.dump(queue, f)


def build_replay(game_env, agent, save=True):
    state = game_env.get_state()
    queue_path = os.path.join(in_dir, 'queue.pickle')
    if os.path.exists(queue_path):
        agent.memory = load_queue()
    else:
        print('running')
        while True:
            episode = []
            for i in range(500):
                action = np.random.choice([0, 1], p=[0.9, 0.1])
                # action = agent.act(state)
                next_state, reward, done, score = game_env.step(action)
                # agent.remember(state, action, reward, next_state, done)
                episode.append((state, action, reward, next_state, done))
                if done:
                    print(score)
                    if score > GOOD_SCORE:
                        agent.memory.extend(episode)
                        print('added good episode, queue size = {}'.format(
                            len(agent.memory)))
                    if np.random.random() < 0.05:
                        agent.memory.extend(episode)
                        print('added bad episode, queue size = {}'.format(
                            len(agent.memory)
                        ))

                state = next_state
        if save:
            save_queue(agent.memory)


def train(episode_count, display):
    # initialize gym environment and the agent
    agent = DQNAgent(2)
    game_env = GameEnv(display)
    build_replay(game_env, agent)

    state = game_env.state
    for e in range(episode_count):
        for time_t in range(500):
            action = agent.act(state)
            next_state, reward, done, score = game_env.step(action)
            # agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print('-' * 50)
                print("episode: {}/{}, score: {} epsilonn: {}".format(
                    e, episode_count, score, agent.epsilon))
                print('-' * 50)
                break
        # train the agent with the experience of the episode
        agent.replay(32)
        if episode_count % 10 == 0:
            agent.save_weights()
        agent.decrease_epsilon(e)


def play():
    agent = DQNAgent(2)
    game_env = GameEnv(display)
    build_replay(game_env, agent, save=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', dest='display', action='store_true')
    parser.set_defaults(display=False)
    args = parser.parse_args()
    print(args)

    display = args.display
    print('display = {}'.format(display))
    if not display:
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    from ple.games.flappybird import FlappyBird
    from ple import PLE

    # play()
    train(10000000, display)

