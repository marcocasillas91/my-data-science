# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:22:39 2019

@author: Marco Casillas

PREDICCIÓN DE RESULTADOS EN PARTIDOS DE LIGA MX DEL EQUIPO CHIVAS MEDIANTE
MACHINE LEARNING.


Programa 3.
Recibe el archivo .csv de los datos más significativos para poder hacer gráficas de puntos
que ayuden a entender la relación entre las diversas características.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from minepy import MINE

def classify(x, y, clf, name, clf_name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = None)
    clf.fit(x_train,y_train)
    y_pred= np.round(clf.predict(x_test))
    precision = clf.score(x_test,y_test)
    precision_t = clf.score(x_train,y_train)
    print("Acc: ", precision, "  Name:", name, "  Classifier:", clf_name)
    print("AccTrain: ", precision_t, "  Name:", name, "  Classifier:", clf_name)
    print (confusion_matrix(y_test, y_pred))
    return [precision,precision_t,y_pred]
    #return clf

def classifyClean(x, y, clf, name, clf_name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = None)
    clf.fit(x_train,y_train)
    y_pred= np.round(clf.predict(x_test))
    precision = clf.score(x_test,y_test)
    precision_t = clf.score(x_train,y_train)
    #print("Acc: ", precision, "  Name:", name, "  Classifier:", clf_name)
    #print("AccTrain: ", precision_t, "  Name:", name, "  Classifier:", clf_name)
    #print (confusion_matrix(y_test, y_pred))
    return [precision,precision_t,y_pred]

def grafica(x,y,Y,title,xlabel,ylabel):
    #plt.scatter(x, y, color="red")
    plt.scatter(x[Y == 0], y[Y ==0], s = 30, c = 'red')
    plt.scatter(x[Y == 1], y[Y ==1], s = 30, c = 'blue')
    plt.scatter(x[Y == 2], y[Y ==2], s = 30, c = 'green')
    #plt.plot(x, y, color="blue")
    #plt.title("Modelo de Regresión Lineal")
    plt.title(title)
    #plt.xlabel("Posición del empleado")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylabel("Sueldo (en USD)")
    plt.show()
    #print(m[:, 1].max())

#from sklearn.preprocessing import MinMaxScaler

#def graficaNorm(x,y,Y,conjunto,title,xlabel,ylabel):
#    #plt.scatter(x, y, color="red")
#    scaler = MinMaxScaler()
#    scaler.fit(conjunto)
#    xNorm = scaler.transform(conjunto)
##    scaler = MinMaxScaler()
##    scaler.fit(y)
##    yNorm = scaler.transform(y)
#
##    plt.scatter(xNorm[Y == 0], yNorm[Y ==0], s = 30, c = 'red')
##    plt.scatter(xNorm[Y == 1], yNorm[Y ==1], s = 30, c = 'blue')
##    plt.scatter(xNorm[Y == 2], yNorm[Y ==2], s = 30, c = 'green')
#    plt.scatter(xNorm[0][Y == 0], xNorm[1][Y ==0], s = 30, c = 'red')
#    plt.scatter(xNorm[0][Y == 1], xNorm[1][Y ==1], s = 30, c = 'blue')
#    plt.scatter(xNorm[0][Y == 2], xNorm[1][Y ==2], s = 30, c = 'green')
#    #plt.plot(x, y, color="blue")
#    #plt.title("Modelo de Regresión Lineal")
#    plt.title(title)
#    #plt.xlabel("Posición del empleado")
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    #plt.ylabel("Sueldo (en USD)")
#    plt.show()
#    #print(m[:, 1].max())
#
#
##dataset = pd.read_csv("futbol1.csv")
##df = pd.read_csv("futbol1.csv")
##dataset = pd.read_csv("Futbol ML/export_futbolRed1.csv")
##df = pd.read_csv("Futbol ML/export_futbolRed1.csv")
##dataset2 = pd.read_csv("Futbol ML/KNN20.csv")
##df2=pd.read_csv("Futbol ML/KNN20.csv")

#Debug
dataset = pd.read_csv("export_futbolRed1.csv")
df = pd.read_csv("export_futbolRed1.csv")
dataset2 = pd.read_csv("KNN20.csv")
df2=pd.read_csv("KNN20.csv")



x = df.iloc[0:336,:-1].values
x1 = df.iloc[0:336,:-1]
y = df.iloc[0:336, -1].values
y1 = df.iloc[0:336, -1]

scaler = MinMaxScaler()
scaler.fit(x)
xNorm = scaler.transform(x)

grafica(df['Dirigidos'],df['Distancia'],y,'Relacion Dirigidos-Distancia','Dirigidos','Distancia')
grafica(df['Dirigidos'],df['G_PPORJ2'],y,'Relacion Dirigidos-PuntosPorPartido General 2 años','Dirigidos','PuntosPorPartido General')
grafica(df['Dirigidos'],df['DiffEdadProm'],y,'Relacion Dirigidos-DiffEdadProm','Dirigidos','DiffEdadProm')


grafica(df['Dirigidos'],df['DiffEdadProm'],y,'Relacion Dirigidos-DiffEdadProm','Dirigidos','DiffEdadProm')
grafica(df['G_DIFGOL3'],df['G_DIFGOL2'],y,'Relacion G_DIFGOL3-DIFGOL2','G_DIFGOL3','G_DIFGOL2')
grafica(df['PosLoc'],df['esLocal'],y,'Relacion PosLoc-PosVis','PosLoc','esLocal')
grafica(df['Dirigidos'],df['G_PUNTOS3'],y,'Relacion Dirigidos-GPuntos3','Dirigidos','GPuntos3')
grafica(df['G_PUNTOS2'],df['G_PUNTOS3'],y,'Relacion GPuntos2-GPuntos3','GPuntos2','GPuntos3')
grafica(df['DT_PG'],df['DT_PP'],y,'Relacion DT_PG-DT_PP','DT_PG','DT_PP')
grafica(df['DT_PP'],df['DT_PE'],y,'Relacion DT_PP-DT_PE','DT_PP','DT_PE')
grafica(df['DifEdad'],df['IDTempo'],y,'Relacion DifEdad-IDTempo','DifEdad','IDTempo')
grafica(df['Dirigidos'],df['IDTempo'],y,'Relacion DifEdad-IDTempo','DifEdad','IDTempo')
grafica(df['IDTempo'],df['Dirigidos'],y,'Relacion IDTempo-Dirigidos','IDTempo','Dirigidos')

grafica(df['G_DIFGOL3'],df['LV_DIFGOL3'],y,'Relacion G_DIFGOL3-LV_DIFGOL3','G_DIFGOL3','LV_DIFGOL3')
grafica(df['IDTempo'],df['Dirigidos'],y,'Relacion IDTempo-Dirigidos','IDTempo','Dirigidos')
grafica(df['IDTempo'],df['Dirigidos'],y,'Relacion IDTempo-Dirigidos','IDTempo','Dirigidos')


#grafica(df[''],df[''],'Relacion ','','')
#grafica(df[''],df[''],'Relacion ','','')
#
## Aplicar k-fold cross validation
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#accuracies.mean()
#accuracies.std()
