#TP2: FeedForwardNeuralNetwork:
#Dataset==données breast_cancer:
import time
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizer_v1 import SGD
from keras.layers import Input, Dense, Activation
from keras.layers import Input, Dense, Flatten, Concatenate, Conv1D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
#Récupérer le dataset:
data=load_breast_cancer()
print(data)
#Récupérer les données du dataset
X=data.data
print(X)
y= data.target
print(y)
#une fonction afin de normaliser les données récupérés du dataset
def normalized(x) :
  return (x- np.mean(x,axis=0))/(np.max(x, axis=0)-np.min(x, axis=0))
#Normaliser les données X
x_nom = normalized(X)

#premier itératon
#x_train, x_test, y_train, y_test = train_test_split(x_nom, y, test_size=0.3, train_size=0.7, random_state=True)
#2ème itération// ajouter les données de validation
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, train_size=0.5, random_state=True)
learning_rate = 0.8
batch_size = 50
nb_epoch = 50
#L'apprentissage du modèle et construction du réseau de neurone
learning_rate = 0.2
batch_size = 50
nb_epoch = 100
#Construction du réseau
model=Sequential()

model.add(Dense(128, input_dim=30))
model.add(Activation('sigmoid'))#  Activation relu
model.add(Dropout(0.3))
#ajouter la fonction de normalisation
model.add(BatchNormalization())
#couche caché
#model.add(Dense(512))
#model.add(Activation('sigmoid'))# Activation relu
#model.add(Dropout(0.3))

#model.add(BatchNormalization())
#enlever une couche caché et voir le résultat
#model.add(Dense(1024))
#model.add(Activation('sigmoid'))
#model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid')) # la couche de sortie toujours avec sigmoid
print(model.summary())
#sauvegarder les données // sauvegarder les bonnes score
checkpointer = ModelCheckpoint(filepath="NN_TP1.hdfs", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
#ajout condition d'arret
early= EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
sgd = SGD(learning_rate)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1,validation_data=(x_valid, y_valid)), callbacks=[checkpointer, early]
#travail à faire
#tableau rapport pour dire la meilleure comparaison en changeant chaque fois un des hyperparameter.
#afin de trouver la meilleur solution.