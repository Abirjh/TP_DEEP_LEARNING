#TP3 : CNN/deeplearning
#dataset= les données manuscrite(mnist)
import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
# 1) load data from mnist dataset
#mnist contient des images manuscrite des chiffres(noir et blanc)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#ajouter les données de validation
x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5, train_size=0.5, random_state=True)
rows, cols, channels = 28,28,1
x_train = x_train.reshape(60000, rows, cols, channels)
x_test = x_test.reshape(5000, rows, cols, channels)
x_validation = x_validation.reshape(5000, rows, cols, channels)
#Etape Pour changer le type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_validation = x_validation.astype('float32')
# Etape pour normaliser les données
# 255  = la taille maximale des données (pixel)
x_train /= 255
x_test /= 255
x_validation /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#Pour vectoriser les données
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)
y_validation=np_utils.to_categorical(y_validation,10)
# Model Architecture
learning_rate = 0.8
#le choix des paramétres impacte sur la qualité de convergence
batch_size = 30
nb_epoch = 50
model = Sequential()
model.add(Conv2D(32,kernel_size=(3, 3),activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64,(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#a la fin des Max Pooling le sortie que des matrices, flatten pour vectoriser les matrices
model.add(Flatten())
#neurone full connected
#128=nbre des neurones
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#10= nbre des classes qu'on a
#la dernière couche tjours (catégorielle)=10
#softmax ** sigmoid à ajouter dans le rapport du résultat
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam', metrics=['accuracy'])
model.summary()
#Entrainer le modèle
model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1,validation_data=(x_validation, y_validation))


