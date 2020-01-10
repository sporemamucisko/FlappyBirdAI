import gym
import matplotlib.pyplot as plt
import numpy as np

from DqnERM import Dqn
from NeuralNetwork import Brain

#Setting parameters
learningRates = 0.001              # Learning Rate
maxMemory = 5000                   # Expierience Replay Memory 5000 expiriences maksymalnie
gamma = 0.9                        # Gamma Discount factor - QValues
batchSize = 32                     # Size of batches inputs and targets
epsilon = 1.                       # Będzie się zmieniać z dzialaniem programu! W tym przypadku wszystkie akcje są randomowe
epsilonDecayRate = 0.995           # Będzie mnożone z esiolnem co grę, żeby się stawał coraz mniejszy

#Initializing the Enviroment, the Brain, and The Eperience Replay Memory
#env = nasze środowisko
brain = Brain(3, 2, learningRates)                       #3 wejścia Pozycja Y Ptaka, Pozycja rury górnej i rury dolnej, 2 wyjścia skok lub nie skok
model = brain.model
DQN = Dqn(maxMemory, gamma)                              #Expirience Replay Memory

#Main loop
epoch = 0
currentState = np.zeros(1,3)                              #Jak wygląda środkowisko przed podjęciem akcji 1 rząd, 3 kolumny
nextState = currentState                                  #Póki co
totRewards = 0                                            #Suma wszystkich Nagród
rewards=list()                                            #TU będą przechowywane nagrody

while True:
    epoch +=1                                             #Za każdym razem kiedy wchodzimy w tą pętle zaczynamy następną epokę

    #Zaczynamy grę
    #env.reset()                                            #Reset giereczki co epokę Metodę reset musimy dodać
    currentState = np.zeros(1, 3)                           #Resetujemy current State i nextstate co epokę
    nextState = currentState
    gameOver = False                                        #Resetujemy na False w nowej epoce
    while not gameOver:

        #Wykonywanie akcji
        if np.random.rand() <= epsilon:                     #Expoloration
           action = np.random.randint(0,2)                  # 0 lub 1 czyli skok lub nie skok
        else:                                               #zachowujemy się tak jak sieć neuronowa
            qValues = model.predict(currentState)[0]        #Generates output predictions for the input samples.
            action = np.argmax(qValues)                     #Wybierze największą qValue

        #Update-owanie enviroment
        # nextState[0], reward, gameOver, _ = env.step(action)       #Metoda step z env pobrać
        # env.render

        #totRewards += reward

        # Remembering new experience, training Ai, updating current state
        #DQN.remember([currentState,action,reward,nextState], gameOver)   #Remebering new expirience, dodajemy do ERM
        inputs, targets = DQN.getBatch(model, batchSize)        #Training AI(batche inputów i outputów)
        model.train_on_batch(inputs,targets)                    #łączy stohastic gradient descend i jakiś inny doczytać

        currentState = nextState                                #Updatuje current State

    # Lowering epsilon and displaying results
    epsilon *= epsilonDecayRate                                 #Zmniejszamy epsilon żeby więcej wybierał a mniej zwiedzał

    print('Epoch '+ str(epoch) + 'Epsilon {:2f}'.format(epsilon) + 'Total reward {:5f}'.format(totRewards))

    rewards.append(totReward)      #Wyświetlanie nagród na wykresie
    totReward = 0
    plt.plot(rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Rewards')
    plt.show()

#env.close()








