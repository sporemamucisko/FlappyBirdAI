#Experience Replay Memory
import numpy as np
#Służy nam to do zapisywania decyzji i uczenia się potem na nich
#Ważne żeby były nie tylko dobre decyzje żeby było wiadomo co jest robione dobrze a co źle

class Dqn():

    def __init__(self, maxMemory, discount):
        self.maxMemory = maxMemory                   #Maksymalna pamięć Experience Replay, żeby nie była za duża
        self.discount = discount                     #Gamma używana do liczenia Q
        self.memory = list()                         #Tu będziemy zapiywać nasze experience

    #Zapamiętywanie nowych experience, dodaje je do naszego memory list
    #Experience przeechowuje Aktualny stan, podjętą akcję, nagrodę i przyszły stan + czyPrzegrana gra
    def remember(self,transition, gameOver):         #Transition To jest w zasazdzie większość experience, czyli Aktualny stan, podjętą akcję, nagrodę i przyszły stan
        self.memory.append([transition, gameOver])   #powiększamy naszą listę naszymi exprerience, które składają się z tranistion + gameOver
        if len(self.memory) > self.maxMemory:
            del self.memory[0]                       #Sprawdzamy czy nasza ilość naszych experience nie jest większa od maksymalnej pamięci, jeżeli tak to usuwamy najstarszą experience

    #Gettings Batches of inputs and targets
    def GetBatch(self, model, batchsize):                     #Do policzenia Qvalues aktualnego stanu i przyszłego stanu
        lenMemory=len(self.memory)
        numInputs=self.memory[0][0][0].shape[1]               #Sięga do naszej pamięci[0] potem do tranisition [0] potem do aktualnego stanu [0], [1] to liczba kolumn, czyli liczba inputów
        numOutputs=model.output_shape[-1]                     #Sięga do liczby neuronów na ostatniej lini, czyli liczby ouputów

        #Initializing inputs and outputs
        inputs=np.zeros((min(batchsize,lenMemory),numInputs))                #Z tego co rozumiem to inputs to jest jakaś liczba current statów wyciągnięta z memory(rzędy) kolumny to wejścia
        targets = np.zeros((min(batchsize,lenMemory),numOutputs))            #Rzędy to Q values przewidziane przez sieć neuronową rzędy to podział na wyjścia

        # Extracting transitions from random experiences                       #enumerate tworzy nową listę dodając tym indexom np 8,2,4 indexy 0,1,2 i będzie iterować po tych drugich inx po pierwsysch
        for i,inx in enumerate(np.random.randint(0,lenMemory, size=min(batchsize,lenMemory))):        #Randomowe indexy z Memory, żeby zabrać randomowe transitions z memory, wielkości batchsize lub jak większe to lenMemory
            currentState, action, reward, nextState = self.memory[inx][0]                             #Wyciągamy tranistion, 0 ponieważ jest pierwsze w memory[transition,gameover]
            gameOver = self.memory[inx][1]                                                            #Wyciągamy gameOver

            # Updating inputs and targets                 Dodajemy current staty i liczymy ich q values
            inputs[i] = currentState
            targets[i] = model.predict(currentState)[0]   #Generates output predictions for the input samples. Czyli genertuje predykcje Qvalues na podstawie currentState
            if gameOver:
                targets[i][action] = reward               #Jeżeli przegramy to Qvalue akcji która do tego doprowadziła będzie = reward
            else:
                targets[i][action] = reward + self.discount * np.max(model.predict(nextState)[0])            #model.predict - Największa wartość Q przewidziana przez nasz model całość to nasza aktualizacja Q
        return inputs,targets                     






