#TP4: CNN
#les données cifar10
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets
from keras.utils import np_utils
#téléchager les données:
(X_train,Y_train),(X_test,Y_test)=datasets.cifar10.load_data()
#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3, train_size=0.7, random_state=True)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_test, Y_test, test_size=0.5, train_size=0.5, random_state=True)
print(X_train.shape)
#Y_train, Y_test, Y_validation sont de 2D array on va faire la transformation en 1D pour la classification
Y_train = Y_train.reshape(-1,)
Y_test = Y_test.reshape(-1,)
Y_validation = Y_validation.reshape(-1,)
#afin de mieux visualizer l'image
plt.figure(figsize=(15,2))
plt.imshow(X_train[0])
#Normaliser les données
X_train=X_train/255
X_test=X_test/255
Y_train=Y_train/255
Y_test=Y_test/255
Y_validation=Y_validation/255
X_validation=X_validation/255
#Pour la vectorisation
Y_train=np_utils.to_categorical(Y_train,10)
Y_test=np_utils.to_categorical(Y_test,10)
Y_validation=np_utils.to_categorical(Y_validation,10)
# Model Architecture
#learning_rate = 0.8(selement lorsqu'on utilise sgd comme optimizer)
#le choix des paramétres impacte sur la qualité de convergence
batch_size = 30
nb_epoch = 2
model = Sequential()
model.add(Conv2D(32,kernel_size=(3, 3),activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#a la fin des Max Pooling la sortie que des matrices, flatten pour vectoriser les matrices
model.add(Flatten())
#neurone full connected
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
#10= nbre des classes qu'on a
#la dernière couche tjours (catégorielle)=10
#softmax ** sigmoid à ajouter dans le rapport le résultat
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
model.summary()
model.fit(X_train,Y_train, epochs=nb_epoch,validation_data=(X_validation, Y_validation))
#Pour l'évaluation de la classification
model.evaluate(X_test,Y_test)
#les données à prédir
predict_x=model.predict(X_test)
classes_x=np.argmax(predict_x,axis=1)
#Rappel, précision, fmesure
print("Rapport de la classification: \n", classification_report(test_label, classes_x))
#matrice de confusion
cm = confusion_matrix(test_label, classes_x)
print("la matrice de confusion: \n", cm)