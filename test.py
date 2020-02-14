from NeuralNetwork import Brain
from gameEnviroment3 import Enviroment
import numpy as np




waitTime = 40
filepathToOpen='model1.h5'
env = Enviroment(waitTime)
brain = Brain(20, 2, 0.001)                       #3 wejścia Pozycja Y Ptaka, Pozycja rury górnej i rury dolnej, 2 wyjścia skok lub nie skok
model = brain.loadModel(filepathToOpen)


currentState = np.zeros((1,20))                              #Jak wygląda środkowisko przed podjęciem akcji 1 rząd, 3 kolumny
nextState = currentState    


while True:
    currentState = np.zeros((1, 20))                           #Resetujemy current State i nextstate co epokę
    nextState = currentState
    gameOver = False      
    while not gameOver:
        
        
        qValues = model.predict(currentState)[0]        #Generates output predictions for the input samples
        action = np.argmax(qValues)
        
        nextState[0], reward, gameOver, _ = env.step(action)
        
        currentState = nextState                                #Updatuje current State
