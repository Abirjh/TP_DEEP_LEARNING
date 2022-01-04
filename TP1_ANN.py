#TP1: FeedForwardNeuralNetwork:
#Dataset==données mnist:
import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Activation
from keras.optimizer_v1 import SGD
# 1) load data from mnist dataset
#mnist contient des images manuscrite des chiffres(noir et blanc)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# représenter les données mnist
plt.figure(figsize=(7.195,3.841), dpi=100)
for i in range(100):
  plt.subplot(10,10,i+1)
  plt.imshow(x_train[i].reshape([28,28]), cmap='gray')
  plt.axis('off')
#Préparer les données pour l'apprentissage
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
#Etape Pour changer le type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Etape pour normaliser les données
# 255  = la taille maximale des données (pixel)
x_train /= 255
x_test /= 255
# la structure d'un réseau de neurone simple et basique
#procéder à l'étape d'apprentissage
learning_rate = 0.8
#le choix des paramétres impacte sur la qualité de convergence
batch_size = 30
nb_epoch = 50
#1ere ligne pour l apprentissage// framework pour le travel //variable de modèle
model = Sequential()
#couche simple //1 : nbre de neurone // dimension des input 28*28 // couche entrée
model.add(Dense(30, input_dim=784))
#une couche d'activation
#entre chaque couche il faut faire une couche d'activation
model.add(Activation('sigmoid'))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#une couche cachée // couche sortie// 1 une sortie possible
model.add(Dense(1, activation='sigmoid'))
#optimizer
sgd= SGD(learning_rate)
#pour compiler
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
#là où l'apprentissage commence
#nb_epoch == le nbre d'apprentissage
model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)
