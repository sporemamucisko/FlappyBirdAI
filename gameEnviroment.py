import numpy as np
import sys
import random
import pygame
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100 # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])




class Enviroment:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY = 0  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward accleration
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps

        self.next_pipe_x = self.lowerPipes[0]['x'] - PIPE_WIDTH  # wartość x rury najbliższej
        self.next_pipe_hole_y = (self.lowerPipes[0]['y'] + (self.upperPipes[0]['y'] + IMAGES['pipe'][0].get_height())) / 2  # wartość y dziury w rurze
        self.next_pipe_bottom = self.lowerPipes[0]['y']
        self.next_pipe_upper = self.upperPipes[0]['y'] + IMAGES['pipe'][0].get_height()


    def step(self, action):
        pygame.event.pump()
        #self.playery, self.next_pipe_x, self.next_pipe_hole_y = self.state


        reward = 0.1                   #Nagroda za przeżycie
        isGameOver = False              #Czy przegranko

        if action == 1:                                       #Jeżeli 1 to skok
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        playerMidPos = self.playerx + PLAYER_WIDTH / 2             #Sprawdzamy czy przeskoczył rurę
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                # SOUNDS['point'].play()
                reward = 5                                         #Jeżeli tak to nagroda duża


        if (self.loopIter + 1) % 3 == 0:                       #Poruszanie się ziemi
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)


        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:   #Poruszanie się gracza
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0


        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):             #Poruszanie się rur
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

            # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

            # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

            # check if crash here
        isCrash = checkCrash({'x': self.playerx, 'y': self.playery,
                            'index': self.playerIndex},
                             self.upperPipes, self.lowerPipes)
        if isCrash or self.playery < 1:
            # SOUNDS['hit'].play()
            # SOUNDS['die'].play()
            isGameOver = True
            self.__init__()
            reward = -2

            # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        # showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))



            # Tu trzeba nasze observation space ogarnąć
        self.state = 0
        self.next_pipe_x = self.lowerPipes[0]['x'] - PIPE_WIDTH  # wartość x rury najbliższej
        self.next_pipe_hole_y = (self.lowerPipes[0]['y'] + (self.upperPipes[0]['y'] + IMAGES['pipe'][0].get_height())) / 2  # wartość y dziury w rurze
        self.next_pipe_bottom = self.lowerPipes[0]['y']
        self.next_pipe_upper = self.upperPipes[0]['y'] + IMAGES['pipe'][0].get_height()

        ###Normalizacja
        height = self.playery
        height = min(SCREENHEIGHT, height) / SCREENHEIGHT - 0.5
        nextPipe = self.next_pipe_x
        nextPipe = nextPipe / 450 - 0.5 # Max pipe distance from player will be 450
        pipe_height = self.next_pipe_hole_y
        pipe_height = min(SCREENHEIGHT, pipe_height) / SCREENHEIGHT - 0.5
        nextBottom = self.next_pipe_bottom
        nextBottom = min(SCREENHEIGHT, nextBottom) / SCREENHEIGHT - 0.5
        nextUpper = self.next_pipe_upper
        nextUpper = min(SCREENHEIGHT, nextUpper) / SCREENHEIGHT - 0.5

        self.state = (height, nextPipe,pipe_height, nextBottom, nextUpper)
        #print(self.state, action)

        e_d1 = pygame.rect.Rect(self.playerx, self.playery, 2, SCREENHEIGHT - self.playery)
        pygame.draw.rect(SCREEN, (255, 0, 0), e_d1)

        e_d3 = pygame.rect.Rect(self.playerx, self.playery, self.next_pipe_x, 2)
        pygame.draw.rect(SCREEN, (255, 0, 0), e_d3)

        e_d4 = pygame.rect.Rect(self.next_pipe_x, self.next_pipe_hole_y, PIPE_WIDTH, 5)
        pygame.draw.rect(SCREEN, (255, 0, 0), e_d4)

        e_d5 = pygame.rect.Rect(self.next_pipe_x, self.next_pipe_bottom , PIPE_WIDTH, 5)
        pygame.draw.rect(SCREEN, (255, 0, 0), e_d5)

        e_d6 = pygame.rect.Rect(self.next_pipe_x, self.next_pipe_upper, PIPE_WIDTH, 5)
        pygame.draw.rect(SCREEN, (255, 0, 0), e_d6)


        pygame.display.update()
        FPSCLOCK.tick(FPS)
            # print ("FPS" , FPSCLOCK.get_fps())
            # print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)

        return self.state, reward, isGameOver, {}


    def reset(self):
        self.clock = pygame.time.Clock()
        reward = 0                  #Nagroda za przeżycie
        isGameOver = False 
        
        self.state, reward, isGameOver, _ = self.step(0)
        return self.state



    #def render(self):




def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False


