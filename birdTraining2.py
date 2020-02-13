import matplotlib.pyplot as plt
import numpy as np

from DqnERM import Dqn
from NeuralNetwork import Brain
from gameEnviroment3 import Enviroment

#Setting parameters
waitTime=0
learningRates = 0.001              # Learning Rate
maxMemory =  50000               # Expierience Replay Memory 5000 expiriences maksymalnie
gamma = 0.90                        # Gamma Discount factor - QValues
batchSize = 32                   # Size of batches inputs and targets
epsilon = 0.8                       # Będzie się zmieniać z dzialaniem programu! W tym przypadku wszystkie akcje są randomowe
epsilonDecayRate = 0.9998        # Będzie mnożone z esiolnem co grę, żeby się stawał coraz mniejszy
epsilonFinal = 0.01

nScore = 0
maxnScore = 0
totnScore = 0
scores=list()
bestscorelist=list()

filepathToSave = 'model1.h5'

#Initializing the Enviroment, the Brain, and The Eperience Replay Memory
env = Enviroment(waitTime)
brain = Brain(20, 2, learningRates)                       #3 wejścia Pozycja Y Ptaka, Pozycja rury górnej i rury dolnej, 2 wyjścia skok lub nie skok
model = brain.model
DQN = Dqn(maxMemory, gamma)                              #Expirience Replay Memory

#Main loop
epoch = 0
currentState = np.zeros((1,20))                              #Jak wygląda środkowisko przed podjęciem akcji 1 rząd, 3 kolumny
nextState = currentState                                  #Póki co
totRewards = 0                                            #Suma wszystkich Nagród
rewards=list()                                            #TU będą przechowywane nagrody

while True:
    epoch +=1                                             #Za każdym razem kiedy wchodzimy w tą pętle zaczynamy następną epokę

    #Zaczynamy grę
    #env.reset()                                            #Reset giereczki co epokę Metodę reset musimy dodać
    currentState = np.zeros((1, 20))                           #Resetujemy current State i nextstate co epokę
    nextState = currentState
    gameOver = False                                        #Resetujemy na False w nowej epoce
    while not gameOver:

        
        randomNumber = np.random.rand()
        #Wykonywanie akcji
        if randomNumber <= epsilon:                     #Expoloration
           action = np.random.randint(0,2)                  # 0 lub 1 czyli skok lub nie skok
        else:                                               #zachowujemy się tak jak sieć neuronowa
            qValues = model.predict(currentState)[0]        #Generates output predictions for the input samples
            action = np.argmax(qValues)                     #Wybierze największą qValue
            #print(qValues)
        
        
        #Update-owanie enviroment
        nextState[0], reward, gameOver, _ = env.step(action)       #Metoda step z env pobrać
        #env.render()
        totRewards += reward
        #print(currentState, nextState)


        # Remembering new experience, training Ai, updating current state
        DQN.remember([currentState,action,reward,nextState], gameOver)   #Remebering new expirience, dodajemy do ERM
        inputs, targets = DQN.GetBatch(model, batchSize)        #Training AI(batche inputów i outputów)
        model.train_on_batch(inputs,targets)                    #łączy stohastic gradient descend oraz Backpropagation na raz Funkcja straty liczona z q values przzewidzianych a takich jakie powinny być według równania belllmana
        
        
        if env.pScore > 0:
            nScore +=1
            
        currentState = nextState                                #Updatuje current State
        

    # Lowering epsilon and displaying results
    epsilon *= epsilonDecayRate                                 #Zmniejszamy epsilon żeby więcej wybierał a mniej zwiedzał
    epsilon = max(epsilonFinal, epsilon)
    
    
    if nScore> maxnScore and nScore>3:
        model.save(filepathToSave)
        maxnScore = nScore
        
##################################################################################Wyniki
    
    print('Epoch '+ str(epoch) + '    Score: ' + str(nScore))

    bestscorelist.append(nScore)      #Wyświetlanie nagród na wykresie
    if epoch % 500 == 0:
        plt.plot(bestscorelist)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.show()
    
    
    totnScore += nScore
    nScore = 0

    
    if epoch % 250 == 0 and epoch!= 0:
        scores.append(totnScore/500)
        totnScore = 0
        plt.plot(scores)
        plt.xlabel('Epoch / 250')
        plt.ylabel('Average Score')
        plt.show()
    

    print('Current Best:' + str(maxnScore) + '    Epsilon {:.3f}'.format(epsilon) + '    Total reward {:.3f}'.format(totRewards))


    rewards.append(totRewards)      #Wyświetlanie nagród na wykresie
    totRewards = 0
    if epoch % 500 == 0:
        plt.plot(rewards)
        plt.xlabel('Epoch')
        plt.ylabel('Rewards')
        plt.show()









