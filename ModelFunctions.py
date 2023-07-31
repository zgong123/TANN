from pandas.core.frame import DataFrame
from pandas import concat
from CreateImageModels import CreateModel1_7
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from numpy import argmax
from sklearn.metrics import f1_score, confusion_matrix
from re import sub
from time import time
from CommonGZ import GetRunningTime
from os.path import join 

def TrainModel(XData, yData, ImageSize = (6, 5, 2), BatchSize = 8, NumberOfEpochs = 120,  LearningRate = 0.0005, NumberOfClasses = 3, Verbose = 0):

    RandomSeed = 1

    Model = CreateModel1_7(ImageSize, NumberOfClasses, RandomSeed)
    Model.compile(loss=CategoricalCrossentropy(label_smoothing = 0.9), optimizer = Adam(learning_rate = LearningRate), metrics=['acc'])
    #print(model.summary())

    XTrain = XData
    yTrain = to_categorical(yData, NumberOfClasses, dtype="int32")

    t0 = time()
    Model.fit(XTrain, yTrain, batch_size = BatchSize, epochs = NumberOfEpochs, verbose = Verbose, shuffle = True)
    t1 = time()

    print("Model Training Finish! ")
    print(GetRunningTime(t0 = t0, t1 = t1))
    return Model
