import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Brain Class
class Brain():

    def __init__(self, numInputs, numOutputs, lr):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.learningRate = lr

        # Creating the neural network
        self.model = Sequential()                                                            #Pusta sieć/struktura do której się dodaje linie -> inizcjalizuje

        self.model.add(Dense(units=32, activation='relu', input_shape=(self.numInputs,)))    #Dodajemy hidden layer, w input shape podajemy liczbę inputów

        self.model.add(Dense(units=16, activation='relu'))                                   #Dodatkowa hidden layer

        self.model.add(Dense(units=self.numOutputs))                                         #Bez aktywacji bo to ostatnia warstwa(będzie liniowa funkcja aktywacji, może dać tutaj TANH lub tą od 0 do 1)

        self.model.compile(optimizer=Adam(lr=self.learningRate), loss='mean_squared_error')  #Optimizer(back propagation algorytm) oraz liczenie funkcji straty, też zobaczyć która będzie najlepsza