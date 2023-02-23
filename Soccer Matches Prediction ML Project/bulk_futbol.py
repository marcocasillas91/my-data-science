# -*- coding: utf-8 -*-
"""
SOCCER MATCHES PREDICTION USING MACHINE LEARNING.
Created on Tue Sep 10 17:08:44 2019
Last Updated: February 10th, 2023
    
Football/Soccer Matches Prediction Machine Learning project
League: Liga MX
Country: Mexico
Team: Guadalajara

@author: Marco Casillas

Program #1

This program receives a .csv file with football statistics; transform such data, finds
most significant variables and returns a subset of data using those variables as a .csv file.
In Program number two, classifier models will be applied using that data.

-Data transformation examples:
    -Deleting irrelevant categorical data
    -Handling missing values
    -Fixing data types and ensuring data consistency (For instance, avoid different names when referring to a single team)
    -Applying maximal Information Coefficient to find most significant variables

-Mutual Information and Wrapper techniques were used at a certain point, but did not
raise the predictions accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from minepy import MINE

#Function to get a classifier accuracy after dividing data in test and train subdatasets
def classify(x, y, clf, name, clf_name, printToConsole):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = None)
    clf.fit(x_train,y_train)
    y_pred= np.round(clf.predict(x_test))
    precision = clf.score(x_test,y_test)
    precision_t = clf.score(x_train,y_train)
    if printToConsole:
        print("Acc: ", precision, "  Name:", name, "  Classifier:", clf_name)
        print("AccTrain: ", precision_t, "  Name:", name, "  Classifier:", clf_name)
        print (confusion_matrix(y_test, y_pred))

#    return [precision,y_pred]
    return [precision,y_pred]
    #return clf



'''
Importing .csv files
    - futbol6.csv is the initial data collection
    - futbol6_a.csv is a most complete data collection  
Dataset and Dataset1 are used as a reference to the dataset previous to the transformation to be made
'''

dataset = pd.read_csv("futbol6.csv")
df = pd.read_csv("futbol6.csv")
dataset1 = pd.read_csv("futbol6_a.csv")
df1 = pd.read_csv("futbol6_a.csv")

#Local Machine Run
'''
dataset = pd.read_csv("Futbol ML/futbol6.csv")
df = pd.read_csv("Futbol ML/futbol6.csv")
dataset1 = pd.read_csv("Futbol ML/futbol6_a.csv")
df1 = pd.read_csv("Futbol ML/futbol6_a.csv")
'''


# Deleting irrelevant categorical data 

def columnDelete(dataframe, listOfColumns):
    for col in listOfColumns:
        del dataframe[col]

def firstTransform(df):
    #Converting relevant categorical variables to a numeric format
    # Loss(D) = 0, Draw(E)= 1, Win(V) = 2 
    df['Resultado']=df["Resultado"].replace(['D','E','V'],[0,1,2])  
    df['PPasado']=df["PPasado"].replace(['D','E','V'],[0,1,2])
    df['ADPPasado']=df["ADPPasado"].replace(['P','E','G'],['D','E','V'])
    df['ADPPasado']=df["ADPPasado"].replace(['D','E','V'],[0,1,2])

    df['Resultado']=np.round(df['Resultado'],decimals=0)
    df['PPasado']=np.round(df['PPasado'],decimals=0)
    #df['ADPPasado']=np.round(df['ADPPasado'],decimals=0)

    #Filling missing info.
    df['Resultado'] = df['Resultado'].fillna(1.0)
    df['PPasado'] = df['PPasado'].fillna(1.0)
    df['ADPPasado'] = df['ADPPasado'].fillna(1.0)

    # Numeric format for variable 'DT'.
    labelencoder_x = LabelEncoder()
    df['DT'] = labelencoder_x.fit_transform(df['DT'])


irrelevantColumns=["Temporada", "Torneo", "Fecha", "Hora", "Local", "Visitante",\
                   "Estadio", "FechaComp", "Jornada", "G_ESP3", "G_ESP2",\
                   "G_ESP1", "LV_ESP3", "LV_ESP2", "LV_ESP1", "GolLocal", \
                   "GolVis", "GF", "GC","IDJuego", "AD_REG5", "AD_REG4", \
                   "AD_REG3", "AD_REG2"]


columnDelete(df,irrelevantColumns)
columnDelete(df1,irrelevantColumns)
firstTransform(df)
firstTransform(df1)

df = df.fillna(df.mean())
df1 = df.fillna(df1.mean())

#Reducing dataset, since the first half of the dataset does not have previous records of rival teams.
#Deleting rows with a lot of missing info

dfRed=df.iloc[182:336,:]
dfRed1=df1.iloc[182:336,:]

#Change export route if needed. Keeping Local Machine path
export_csv = df.to_csv (r'D:\Archivos\Machine Learning Centraal\Futbol ML\export_futbol.csv', index = None, header=True) 
export_csv = dfRed1.to_csv (r'D:\Archivos\Machine Learning Centraal\Futbol ML\export_futbolRed1.csv', index = None, header=True)

x = dfRed1.iloc[0:336,:-1].values
x1 = dfRed1.iloc[0:336,:-1]
y = dfRed1.iloc[0:336, -1].values
y1 = dfRed1.iloc[0:336, -1]


#p, ypred=classify(x, y,KNeighborsClassifier() , 'Prueba Con Fselection', 'KNN',printToConsole = True)
#pTest,Ptrain, ypred=classify(x, y,KNeighborsClassifier() , 'Prueba Con Fselection', 'KNN',printToConsole = True)



#Maximal Information Coefficient technique to extract most significant variables
micList=[]
#for j in range(x2.shape[1]):
for j in range(x.shape[1]):
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x[:,j], y)
    print ("MIC", mine.mic())
    micList.append(mine.mic())
    
micList1=micList.copy()
top20=[]
indexes=[]
#indexes1=[]

numVar=len(micList)
for k in range(numVar):
    ind=micList1.index(max(micList1),0,numVar)
    indexes.append(ind)
    #ind1=micList1.index(max(micList1),0,len(micList))
    #indexes1.append(ind1)
    top20.append(df1.columns[ind])
    micList1[ind] = 0

"""Indices de Maximal Information Coefficient definidos en pruebas anteriores
#Indices de MicList:
#6,59,4,5,58,16,24,7,0,8
    
#Indices de MicList2:
#62,6,59,4,5,58,16,24,67,7,0,8,66,17,106,135,22,68,138,137,60,65...
    
#Indices10: 62,6,59,4,5,58,16,24,67,7
#Indices20: 62,6,59,4,5,58,16,24,67,7,0,8,66,17,106,135,22,68,138,137
#Indices15: 62,6,59,4,5,58,16,24,67,7,0,8,66,17,106

Fin de Indices de Maximal Information Coefficient definidos en pruebas anteriores"""

#Indices de MicList3, de mayor a menor:
#62,24,59,4,16,6,1,5,8,15,23,135,68,67,48,66,137,58,138,17,7,0,25,2,38,40...
    
#Indices10(Top 10): 62,24,59,4,16,6,1,5,8,15
#Indices20(Top 20): 62,24,59,4,16,6,1,5,8,15,23,135,68,67,48,66,137,58,138,17
#Indices15(Top 15): 62,24,59,4,16,6,1,5,8,15,23,135,68,67,48


#Indices de menor a mayor Relevancia :
#82,86,90,95,89,97,3,88,91,94,79,81,78,85,93,162,150,152,153,155,156,161,96,92,
#53,160,148,149,80,84,87,163,157,83,50,64,44,131,72,76,122,52,70,151,125,146,112
#36,158,120



'''
OTHER TESTS PERFORMED. MIC TECHNIQUE WAS THE MOST USEFUL OF THEM ALL.
INDEXES WERE EXTRACTED SO THAT PCA ANALYSIS COULD BE PERFORMED ON 
GRID SEARCH FILE
'''


pca = PCA(n_components=5)
pca.fit(x)
x_pca=pca.transform(x)




scaler = MinMaxScaler()
scaler.fit(x)
x2 = scaler.transform(x)

"""
#Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = None)
"""
# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test  = sc_x.transform(x_test)
"""


def feature_selection(X,y,numf):
    mi = mutual_info_classif
    info = mi(X,y)
    rank = np.argsort(info)
    lrank = list(rank)
    indexes = []
    for i in range(numf):
        f = np.argmax(info)
        indexes.append(f)
        info[f] = 0
    X1 = X[:,indexes]
    return X1,indexes

def rankingFeaturesF(X,y,datatype=0,numf=0):
    if(datatype==0):
        mi = mutual_info_classif
    else:
        mi = mutual_info_regression
    info = mi(X,y)
    info = info/max(info)
    indexes = []
    if(numf<=0):
        numf = X.shape[1]
    for i in range(numf):
        #argmax regresa el indice del valor máximo
        f = np.argmax(info)
        indexes.append(f)
        info[f] = -info[f]
    return [np.abs(info),indexes]

def rankingFeaturesT(X,y,datatype=0,threshold=0.5):
    if(datatype==0):
        mi = mutual_info_classif
    else:
        mi = mutual_info_regression
    info = mi(X,y)
    info = info/max(info)
    indexes = []
    for i in range(X.shape[1]):
        f = np.argmax(info)
        score = info[f]
        if(score>=threshold):
            indexes.append(f)
            info[f] = -info[f]
        else:
            break
    return [np.abs(info),indexes]

"""
#Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.neighbors import KNeighborsClassifier
#Con p = 2, la distancia es euclidea
classifier = KNeighborsClassifier(n_neighbors = 3, metric= "minkowski", p = 2)
classifier.fit(x_train, y_train)

#Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(x_test)

# Elaborar una matriz de confusión para verificar qué tanto coincide la 
# predicción con y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

score=classifier.score(x_test,y_test)
print(score)
print(cm)

"""
classify(x1, y,KNeighborsClassifier() , 'Prueba Sin Fselection', 'KNN',printToConsole = True)
X1,indexes = feature_selection(x,y,25)
classify(X1, y,KNeighborsClassifier() , 'Prueba Con Fselection', 'KNN',printToConsole = True)
#No se logra incrementar la precision mucho con esta tecnica. Profundizar mas
# o usar otro algoritmo


# WRAPPER METHOD FOR FEATURE SELECTION
def wrapper(x,y,clf,name,clf_name,threshold):
    xInit=x.iloc[:, 0:1]
    pAnt,ypred = classify(xInit,y,clf,name,clf_name, printToConsole = False)
    lf=[0]
    for i in range(1,x1.shape[1]):
        lf.append(int(i))
        xTemp = x.iloc[:, lf]
        [pActual,ypred]= classify(xTemp,y,clf,name,clf_name,printToConsole = False)
        if ((pActual- pAnt) >= threshold):
                pAnt= pActual
        else:
            lf.pop()
    #print(lf)
    #print (pActual)
    return [lf,pActual]


lf= wrapper(x1,y1,KNeighborsClassifier(),"Prueba con Wrapper","KNN",0.005)[0]

opT=np.arange(0.001,0.1,0.002)
lpre=[0]
lind=[0]

"""
for i in range(len(opT)):
    lind.append(wrapper(x1,y1,KNeighborsClassifier(),"Prueba con Wrapper","KNN",opT[i])[0])
    lpre.append(wrapper(x1,y1,KNeighborsClassifier(),"Prueba con Wrapper","KNN",opT[i])[1])
np.argmax(lpre)
"""
#Usando diferentes thresholds,se observa que la mejor precision se obtiene
#con las variables 0,1,11,13,26(19 tras NA promedio) y 37, las cuales son:
#IDTempo,PosLoc,G_PJ3,G_PP3,G_PJ1(G_PJ2),LV_PP3
#

lf,p= wrapper(x1,y1,KNeighborsClassifier(),"Prueba con Wrapper","KNN",0.0001)


info,index=rankingFeaturesT(x1,y,datatype=0,threshold=0.5)
info,index=rankingFeaturesF(x1,y,datatype=0, numf=15)

x2 = df.iloc[0:336,[1,7,8,9,15,22,23,24,25,26,27,28,36,37,41,42,46,48,55,57]].values
x_2 = df.iloc[0:336,[1,7,8,9,15,22,23,24,25,26,27,28,36,37,41,42,46,48,55,57]]
y2 = df.iloc[0:336, -1].values
y_2 = df.iloc[0:336, -1]


p,ypred=classify(x2, y2,KNeighborsClassifier() , 'Prueba Con Fselection', 'KNN',printToConsole = True)
p,ypred=classify(x, y,RandomForestClassifier(max_depth=10,n_estimators=25) , 'Prueba Con Fselection', 'RandomForest',printToConsole = True)




