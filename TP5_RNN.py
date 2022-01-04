import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import pandas as pd
from sklearn.metrics import confusion_matrix , classification_report
#importer les données
y=pd.read_csv(r'D:\tp5_RNN\y_train.txt',delimiter='\s+', header=None)
X1=pd.read_csv(r'D:\tp5_RNN\X_train.txt',delimiter='\s+', header=None)
X_test=pd.read_csv(r'D:\tp5_RNN\X_test.txt',delimiter='\s+', header=None)
y_test=pd.read_csv(r'D:\tp5_RNN\y_test.txt',delimiter='\s+', header=None)
#Transformer en np array
X_train_mat= np.asarray(X_train)
print(X_train_mat)
print(X_train_mat.shape)
y_train_mat= np.asarray(y_train)
X_test_mat= np.asarray(X_test)
y_test_mat= np.asarray(y_test)
#fonction de normalisation
def normalized(x) :
    return (x- np.mean(x,axis=0))/(np.max(x, axis=0)-np.min(x, axis=0))
#normaliser les données
x_nom = normalized(X_train_mat)
y_nom= normalized(y_train_mat)
x_nom_test= normalized(X_test_mat)
y_nom_test= normalized(y_test_mat)
#Construction du réseau RNN
model=Sequential()
#couche 1
model.add(LSTM(50, return_sequences=True, input_shape=(x_nom.shape[1],1)))
#couche 2
model.add(LSTM(50,return_sequences=True))
#couche 3
model.add(LSTM(50,return_sequences=True))
#couche 4
model.add(LSTM(50,return_sequences=True))
#couche 5
model.add(LSTM(50,return_sequences=True))
#couche 6
model.add(LSTM(50))
#la couche de sortie
model.add(Dense(1))
#Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])#la fonction cout est mean_squared_error
#représenter le modèle
model.summary()
#fitting model
model.fit(x_nom,y_nom,epochs=10,batch_size=50)
#evaluer les performances du modèle
model.evaluate(x_nom_test,y_nom_test)
predict_x=model.predict(x_nom_test)
classes_x=np.argmax(predict_x,axis=1)
yy=np.array(y_test)
#rappel, précision, fmesure
print("Rapport de la classification: \n", classification_report(yy, classes_x))
#matrice de confusion.
cm = confusion_matrix(yy, classes_x)
print("la matrice de confusion: \n", cm)