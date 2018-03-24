import pygame
import sys
import os

DIR_PATH = os.path.dirname(__file__)

import os, sys

# set SDL to use the dummy NULL video driver,
#   so it doesn't need a windowing system.

def sprite(name):
    return pygame.image.load(
        os.path.join(DIR_PATH, 'assets/sprites/{}'.format(name)))


def audio(name):
    ext = 'wav' if 'win' in sys.platform else 'ogg'
    return pygame.mixer.Sound(
        os.path.join(DIR_PATH, 'assets/audio/{}.{}'.format(name, ext)))


def load():
    PLAYER_PATH = (
            'redbird-upflap.png',
            'redbird-midflap.png',
            'redbird-downflap.png',
    )

    # path of background
    BACKGROUND_PATH = 'background-black.png'

    # path of pipe
    PIPE_PATH = 'pipe-green.png'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # numbers sprites for score display
    IMAGES['numbers'] = (
        sprite('0.png').convert_alpha(),
        sprite('1.png').convert_alpha(),
        sprite('2.png').convert_alpha(),
        sprite('3.png').convert_alpha(),
        sprite('4.png').convert_alpha(),
        sprite('5.png').convert_alpha(),
        sprite('6.png').convert_alpha(),
        sprite('7.png').convert_alpha(),
        sprite('8.png').convert_alpha(),
        sprite('9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = sprite('base.png').convert_alpha()

    SOUNDS['die'] = audio('die')
    SOUNDS['hit'] = audio('hit')
    SOUNDS['point'] = audio('point')
    SOUNDS['swoosh'] = audio('swoosh')
    SOUNDS['wing'] = audio('wing')

    # select random background sprites
    IMAGES['background'] = sprite(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = (
        sprite(PLAYER_PATH[0]).convert_alpha(),
        sprite(PLAYER_PATH[1]).convert_alpha(),
        sprite(PLAYER_PATH[2]).convert_alpha(),
    )

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            sprite(PIPE_PATH).convert_alpha(), 180),
        sprite(PIPE_PATH).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, SOUNDS, HITMASKS

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask
