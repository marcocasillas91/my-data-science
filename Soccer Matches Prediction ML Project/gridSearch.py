# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:08:44 2019
Last Updated: February 10th, 2023

Football/Soccer Matches Prediction Machine Learning project
League: Liga MX
Country: Mexico
Team: Guadalajara

@author: Marco Casillas

Program #2

This program receives cleaned and transformed data from file "futbol.py" as a CSV file.
Then, it creates subsets of the input data using the 10, 15 and 20 most significant 
independent variables with the intention of finding the best data subset and the best model.

Principal Component Analysis (PCA) technique is then applied to find a transformation
that increases the prediction accuracy.

Finally, benchmark and Grid Search functions display accuracy results.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics import (
    accuracy_score as acc_score,
    recall_score as recall,
    confusion_matrix
)

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from minepy import MINE
from sklearn.preprocessing import MinMaxScaler

#Models
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

#Classifier function that prints prediction sets and test and training accuracy to console
def classify(x, y, clf, name, clf_name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = None)
    clf.fit(x_train,y_train)
    y_pred= np.round(clf.predict(x_test))
    precision = clf.score(x_test,y_test)
    train_precision = clf.score(x_train,y_train)
    print("Acc: ", precision, "  Name:", name, "  Classifier:", clf_name)
    print("AccTrain: ", train_precision, "  Name:", name, "  Classifier:", clf_name)
    print (confusion_matrix(y_test, y_pred))
    return [precision,train_precision,y_pred]
    #return clf

#Classifier function that returns confusion matrix and test and training accuracy. No printing to console
def classifyClean(x, y, clf, name, clf_name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = None)
    clf.fit(x_train,y_train)
    y_pred= np.round(clf.predict(x_test))
    precision = clf.score(x_test,y_test)
    train_precision = clf.score(x_train,y_train)
    cm= confusion_matrix(y_test, y_pred)
    return [precision,train_precision, cm]
    

def classify1(x, y, clf, name, clf_name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = None)
    clf.fit(x_train,y_train)
    y_pred= np.round(clf.predict(x_test))

    precision = clf.score(x_test,y_test)
    train_precision = clf.score(x_train,y_train)
    acc=acc_score(y_test,y_pred)
    rec=recall(y_test,y_pred,average='micro')
    cm= confusion_matrix(y_test, y_pred)
    return [precision,train_precision, cm,acc,rec]

def grafica(m,title,xlabel,ylabel):
    x = m[:, 0]
    y = m[:, 1]
    #plt.scatter(x, y, color="red")
    plt.plot(x, y, color="blue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    print(m[:, 1].max())
    
def grafica1(m,title,xlabel,ylabel):
    x = m[:, 0]
    y = m[:, 1]
    plt.scatter(x, y, color="red")
    #plt.plot(x, y, color="blue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    print(m[:, 1].max())
    
def wrapper(x,y,clf,name,clf_name,threshold):
    xInit=x.iloc[:, 0:1]
    pAnt,ypred = classifyClean(xInit,y,clf,name,clf_name)
    lf=[0]
    for i in range(1,x.shape[1]):
        lf.append(int(i))
        xTemp = x.iloc[:, lf].values
        [pActual,ypred]= classifyClean(xTemp,y,clf,name,clf_name)
        if ((pActual- pAnt) >= threshold):
                pAnt= pActual
        else:
            lf.pop()

    return [lf,pActual]

# Benchmark Classifier Names.
names = [
        "Perceptron",
        #"XGBreglinear",
        #"XGBreglogistic",
        "NearestNeighbors",
        "LinearSVM",
        "RbfSVM", 
        "PolySVM",
        "SigmoidSVM",
        #"DecisionTree",
        #"RandomForest",
        #"AdaBoost",
        "NeuralNet",
        "NaiveBayes"
        #"LDA",
        #"QDA"
        ]

# Benchmark Classifiers with proposed parameters
classifiers = [
        Perceptron(),
        #xgb.XGBClassifier(objective='reg:linear'),
        #xgb.XGBClassifier(objective='reg:logistic'),
        KNeighborsClassifier(n_neighbors=7,weights='uniform'),
        SVC(kernel="linear",max_iter=1000,gamma='scale',tol=0.06,verbose=False,probability=True),
        SVC(kernel="rbf",max_iter=1000,gamma='auto',tol=0.1,verbose=False),
        SVC(kernel="poly",max_iter=1000,gamma='auto',tol=0.1,verbose=False,degree=3),
        SVC(kernel="sigmoid",max_iter=1000,gamma='auto',tol=0.1,verbose=False,degree=3),
        #DecisionTreeClassifier(max_depth=5),
        #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        #AdaBoostClassifier(),
        MLPClassifier(activation='logistic',verbose=False,tol=0.01,learning_rate_init=0.001,learning_rate='adaptive'),
        GaussianNB(),
        #LinearDiscriminantAnalysis(),
        #QuadraticDiscriminantAnalysis()
        ]



# Reading from transformed dataset "export_futbolRed1.csv"
dataset2 = pd.read_csv("export_futbolRed1.csv")
df2=pd.read_csv("export_futbolRed1.csv")

x = df2.iloc[0:336,:-1].values
x1 = df2.iloc[0:336,:-1]
y = df2.iloc[0:336, -1].values
y1 = df2.iloc[0:336, -1]


"""
MAXIMAL INFORMATION COEFFICIENT INDEXES FROM PREVIOUS TESTS
MicList2 Indexes:
#62,6,59,4,5,58,16,24,67,7,0,8,66,17,106,135,22,68,138,137,60,65...
    
#Indexes from 10 most significant variables: 62,6,59,4,5,58,16,24,67,7
#Indexes from 20 most significant variables: 62,6,59,4,5,58,16,24,67,7,0,8,66,17,106,135,22,68,138,137
#Indexes from 15 most significant variables: 62,6,59,4,5,58,16,24,67,7,0,8,66,17,106
#Indices de MicList3:
#62,24,59,4,16,6,1,5,8,15,23,135,68,67,48,66,137,58,138,17,7,0,25,2,38,40...
    
#Indices10: 62,24,59,4,16,6,1,5,8,15
#Indices20: 62,24,59,4,16,6,1,5,8,15,23,135,68,67,48,66,137,58,138,17
#Indices15: 62,24,59,4,16,6,1,5,8,15,23,135,68,67,48
"""

#Mutual info classif Coefficient Subsets using top 10, 15 and 20 most significant variables 
xMic10=df2.iloc[:,[1, 4, 5, 6, 8, 15, 16, 24, 59, 62]]
xMic20=df2.iloc[:,[1, 4, 5, 6, 8, 15, 16, 17, 23, 24, 48, 58, 59, 62, 66, 67, 68, 135, 137, 138]]
#xMic20=df2.iloc[0:336,[62,6,59,4,5,58,16,24,67,7,0,8,66,17,106,135,22,68,138,137]]
xMic15=df2.iloc[:,[1, 4, 5, 6, 8, 15, 16, 23, 24, 48, 59, 62, 67, 68, 135]]
xMic25=df2.iloc[:,[0, 1, 2, 4, 5, 6, 7, 8, 15, 16, 17, 23, 24, 25, 38, 48, 58,\
                    59, 62, 66, 67, 68, 135, 137, 138]]
#mutual_info_classif  Subsets using top 19 and 25 most significant variables 
xMic19=df2.iloc[:,[1, 4, 5, 6, 8, 15, 16, 23, 24, 48, 58, 59, 62, 66, 67, 68, 135, 137, 138]]
xMic21=df2.iloc[:,[1, 4, 5, 6, 7, 8, 15, 16, 17, 23, 24, 48, 58, 59, 62, 66, 67, 68, 135, 137, 138]]



# MIN-MAX Scaling Subsets
scaler = MinMaxScaler()
scaler.fit(xMic10)
xMic10 = scaler.transform(xMic10)
scaler.fit(xMic15)
xMic15 = scaler.transform(xMic15)
scaler.fit(xMic20)
xMic20 = scaler.transform(xMic20)
scaler.fit(xMic25)
xMic25 = scaler.transform(xMic25)
scaler.fit(xMic19)
xMic19 = scaler.transform(xMic19)
scaler.fit(xMic21)
xMic21 = scaler.transform(xMic21)

#NEW COMPONENTS
#PCA GRAPH USING 2 COMPONENTS TO SEPARATE THE RESULTS SOMEHOW.
pca = PCA(n_components=2)
pca.fit(xMic20)
x_pca=pca.transform(xMic20)
plt.scatter(x_pca[:,0],x_pca[:,1], c=y)
plt.show()

#PCA GRAPH USING 3 COMPONENTS FOR CLASSIFYING TEAM'S WIN, LOSS OR DRAW
pca = PCA(n_components=3)
pca.fit(xMic20)
x_pca3=pca.transform(xMic20)
#plt.scatter(x_pca[:,0],x_pca[:,1], c=y)
#plt.show()

#PCA GRAPH USING 4 COMPONENTS
pca = PCA(n_components=4)
pca.fit(xMic20)
x_pca4=pca.transform(xMic20)
#plt.scatter(x_pca[:,0],x_pca[:,1], c=y)
#plt.show()

#PCA GRAPH USING 3 COMPONENTS FOR TOP 15 MOST SIGNIFICANT VARIABLES
pca = PCA(n_components=3)
pca.fit(xMic15)
x_pca3_15=pca.transform(xMic15)
#plt.scatter(x_pca[:,0],x_pca[:,1], c=y)
#plt.show()

#PCA GRAPH USING 3 COMPONENTS FOR TOP 19 MOST SIGNIFICANT VARIABLES
pca = PCA(n_components=3)
pca.fit(xMic19)
x_pca3_19=pca.transform(xMic19)
#plt.scatter(x_pca[:,0],x_pca[:,1], c=y)
#plt.show()

#PCA GRAPH USING 3 COMPONENTS FOR TOP 21 MOST SIGNIFICANT VARIABLES
pca = PCA(n_components=3)
pca.fit(xMic21)
x_pca3_21=pca.transform(xMic21)

# FUNCTION THAT TAKES THREE SUBDATASETS AND MEASURES AVERAGE TEST AND TRAIN ACCURACY FOR 
# CERTAIN NUMBER OF ITERATIONS BY SHOWING AN ACCURACY HISTOGRAM

def compPrecision(x1,x2,x3,y,model,n_iter,modelname,descrip):
    n_iter=500
    precisionesTrainX1=[]
    precisionesTestX1=[]
    precisionesTrainX2=[]
    precisionesTestX2=[]
    precisionesTrainX3=[]
    precisionesTestX3=[]
    confmat=np.zeros([3,3])
    for i in range(n_iter):
        precisionesTestX1.append(classify1(x1, y, model , descrip, modelname)[0])
        precisionesTrainX1.append(classify1(x1, y, model , descrip, modelname)[1])
        precisionesTestX2.append(classify1(x2, y, model , descrip, modelname)[0])
        precisionesTrainX2.append(classify1(x2, y, model , descrip, modelname)[1])
        precisionesTestX3.append(classify1(x3, y, model , descrip, modelname)[0])
        precisionesTrainX3.append(classify1(x3, y, model , descrip, modelname)[1])
        confmatNueva=classify1(x3, y, model , descrip, modelname)[2]
        confmat = confmat + confmatNueva
        

    print(modelname)
    print(" Ptest XMic10: ",np.mean(precisionesTestX1))
    print("AVG PTrain XMic10: ",np.mean(precisionesTrainX1))
    print("AVG Ptest XMic15: ",np.mean(precisionesTestX2))
    print("AVG PTrain XMic15: ",np.mean(precisionesTrainX2))
    print("AVG Ptest XMic20: ",np.mean(precisionesTestX3))
    print("AVG PTrain XMic20: ",np.mean(precisionesTrainX3))
    print(np.round(confmat/n_iter))

    num_bins = 10
    n, bins, patches = plt.hist(precisionesTestX2, num_bins, facecolor='red', alpha=0.2)
    plt.title("Histograma de P test con con Variables Xmic10, 15 y 20 Norm.")
    plt.show()


# FUNCTION THAT TAKES TWO SUBDATASETS AND MEASURES AVERAGE TEST AND TRAIN ACCURACY FOR 
# CERTAIN NUMBER OF ITERATIONS AND THREE CLASSIFIERS BY SHOWING AN ACCURACY HISTOGRAM
def compPrecisionMod(x1,x2,y,model1,model2,model3,n_iter,modelname1,modelname2,modelname3,descrip):
    n_iter=500
    
    precisionesTrainX1Mod1,precisionesTestX1Mod1=[],[]
    precisionesTrainX1Mod2, precisionesTestX1Mod2=[],[]
    precisionesTrainX1Mod3, precisionesTestX1Mod3=[],[]
    precisionesTrainX2Mod1,precisionesTestX2Mod1=[],[]
    precisionesTrainX2Mod2, precisionesTestX2Mod2=[],[]
    precisionesTrainX2Mod3, precisionesTestX2Mod3=[],[]
    
    confmatX1mod1=np.zeros([3,3])
    confmatX2mod1=np.zeros([3,3])
    confmatX1mod2=np.zeros([3,3])
    confmatX2mod2=np.zeros([3,3])
    confmatX1mod3=np.zeros([3,3])
    confmatX2mod3=np.zeros([3,3])
    
    accuracyX1Mod1,accuracyX2Mod1=[],[]
    accuracyX1Mod2,accuracyX2Mod2=[],[]
    accuracyX1Mod3,accuracyX2Mod3=[],[]
    recallX1Mod1, recallX2Mod1=[],[]
    recallX1Mod2, recallX2Mod2=[],[]
    recallX1Mod3, recallX2Mod3=[],[]

    for i in range(n_iter):
        precisionesTestX1Mod1.append(classifyClean(x1, y, model1 , descrip, modelname1)[0])
        precisionesTrainX1Mod1.append(classifyClean(x1, y, model1 , descrip, modelname1)[1])
        precisionesTestX1Mod2.append(classifyClean(x1, y, model2 , descrip, modelname2)[0])
        precisionesTrainX1Mod2.append(classifyClean(x1, y, model2 , descrip, modelname2)[1])
        precisionesTestX1Mod3.append(classifyClean(x1, y, model3 , descrip, modelname3)[0])
        precisionesTrainX1Mod3.append(classifyClean(x1, y, model3 , descrip, modelname3)[1])
        precisionesTestX2Mod1.append(classifyClean(x2, y, model1 , descrip, modelname1)[0])
        precisionesTrainX2Mod1.append(classifyClean(x2, y, model1 , descrip, modelname1)[1])
        precisionesTestX2Mod2.append(classifyClean(x2, y, model2 , descrip, modelname2)[0])
        precisionesTrainX2Mod2.append(classifyClean(x2, y, model2 , descrip, modelname2)[1])
        precisionesTestX2Mod3.append(classifyClean(x2, y, model3 , descrip, modelname3)[0])
        precisionesTrainX2Mod3.append(classifyClean(x2, y, model3 , descrip, modelname3)[1])
        
        confmatNuevaX1m1=classifyClean(x1, y, model1, descrip, modelname1)[2]
        confmatNuevaX2m1=classifyClean(x2, y, model1, descrip, modelname1)[2]
        confmatNuevaX1m2=classifyClean(x1, y, model2, descrip, modelname2)[2]
        confmatNuevaX2m2=classifyClean(x2, y, model2, descrip, modelname2)[2]
        confmatNuevaX1m3=classifyClean(x1, y, model3, descrip, modelname3)[2]
        confmatNuevaX2m3=classifyClean(x2, y, model3, descrip, modelname3)[2]
        
        confmatX1mod1= confmatX1mod1 + confmatNuevaX1m1
        confmatX2mod1= confmatX2mod1 + confmatNuevaX2m1
        confmatX1mod2= confmatX1mod2 + confmatNuevaX1m2
        confmatX2mod2= confmatX2mod2 + confmatNuevaX2m2
        confmatX1mod3= confmatX1mod3 + confmatNuevaX1m3
        confmatX2mod3= confmatX2mod3 + confmatNuevaX2m3
        
        accuracyX1Mod1.append(classify1(x1, y, model1 , descrip, modelname1)[3])
        recallX1Mod1.append(classify1(x1, y, model1 , descrip, modelname1)[4])
        accuracyX1Mod2.append(classify1(x1, y, model2 , descrip, modelname2)[3])
        recallX1Mod2.append(classify1(x1, y, model2 , descrip, modelname2)[4])
        accuracyX1Mod3.append(classify1(x1, y, model3 , descrip, modelname3)[3])
        recallX1Mod3.append(classify1(x1, y, model3 , descrip, modelname3)[4])
        accuracyX2Mod1.append(classify1(x2, y, model1 , descrip, modelname1)[3])
        recallX2Mod1.append(classify1(x2, y, model1 , descrip, modelname1)[4])
        accuracyX2Mod2.append(classify1(x2, y, model2 , descrip, modelname2)[3])
        recallX2Mod2.append(classify1(x2, y, model2 , descrip, modelname2)[4])
        accuracyX2Mod3.append(classify1(x2, y, model3 , descrip, modelname3)[3])
        recallX2Mod3.append(classify1(x2, y, model3 , descrip, modelname3)[4])
        
    
    print("-----------------------------------------")
    print(modelname1)
    print("-----------------------------------------")
    print("\nAVG Ptest X1: ",np.mean(precisionesTestX1Mod1))
    print("AVG PTrain X1: ",np.mean(precisionesTrainX1Mod1))
    print('Confusion matrix model ',modelname1,' for X1')
    print(np.round(confmatX1mod1/n_iter))
    print('Accuracy: ', np.mean(accuracyX1Mod1))
    print('Recall: ', np.mean(recallX1Mod1))
    
    print("\n-----------------------------------------")
    print("\nAVG Ptest X2: ",np.mean(precisionesTestX2Mod1))
    print("AVG PTrain X2: ",np.mean(precisionesTrainX2Mod1))
    print('Confusion matrix model ',modelname1,' for X2')
    print(np.round(confmatX2mod1/n_iter))
    print('Accuracy: ', np.mean(accuracyX2Mod1))
    print('Recall: ', np.mean(recallX2Mod1))
    
    
    print("\n-----------------------------------------")
    print(modelname2)
    print("-----------------------------------------")
    print("\nAVG Ptest X1: ",np.mean(precisionesTestX1Mod2))
    print("AVG PTrain X1: ",np.mean(precisionesTrainX1Mod2))
    print('Confusion matrix modelo ',modelname2,' para X1')
    print(np.round(confmatX1mod2/n_iter))
    print('Accuracy: ', np.mean(accuracyX1Mod2))
    print('Recall: ', np.mean(recallX1Mod2))
    
    print("\n-----------------------------------------")
    print("\nAVG Ptest X2: ",np.mean(precisionesTestX2Mod2))
    print("AVG PTrain X2: ",np.mean(precisionesTrainX2Mod2))
    print('Confusion matrix modelo ',modelname2,' para X2')
    print(np.round(confmatX2mod2/n_iter))
    print('Accuracy: ', np.mean(accuracyX2Mod2))
    print('Recall: ', np.mean(recallX2Mod2))
    
    print("-----------------------------------------\n")

    print(modelname3)
    print("\n-----------------------------------------")

    print("AVG Ptest X1: ",np.mean(precisionesTestX1Mod3))
    print("AVG PTrain X1: ",np.mean(precisionesTrainX1Mod3))
    print('Confusion matrix modelo ',modelname3,' para X1')
    print(np.round(confmatX1mod3/n_iter))
    print('Accuracy: ', np.mean(accuracyX1Mod3))
    print('Recall: ', np.mean(recallX1Mod3))
    print("\n-----------------------------------------")
    print("\nAVG Ptest X2: ",np.mean(precisionesTestX2Mod3))
    print("AVG PTrain X2: ",np.mean(precisionesTrainX2Mod3))
    print('Confusion matrix modelo ',modelname3,' para X2')
    print(np.round(confmatX2mod3/n_iter))
    print('Accuracy: ', np.mean(accuracyX2Mod3))
    print('Recall: ', np.mean(recallX2Mod3))
    #confmat=confmat/n_iter
    #print(np.round(confmat/n_iter))



# BENCHMARK FOR DIFFERENT CLASSIFIERS AND CHARACTERISTICS SUBSETS
for name,clf in zip(names,classifiers):
    print()
    print("--------------------------------")
    print("Accuracy for :", name)
    compPrecision(x_pca,x_pca3,x_pca4,y,clf,500 ,name,'Linear SVC with Mic10,15 y 20 Norm.')

compPrecision(xMic10,x_pca,xMic20,y,KNeighborsClassifier(n_neighbors=5),500,'KNN','KNN checking PCA 2')

# GRID SEARCH FOR KNN CLASSIFIER KNN (WAS FOUND TO BE THE BEST CLASSIFIER FOR THIS CHALLENGE)
compPrecisionMod(x_pca3,x_pca3_19,y,KNeighborsClassifier(n_neighbors=5),\
                 KNeighborsClassifier(n_neighbors=7),KNeighborsClassifier(n_neighbors=11),\
                 500,'KNN5','KNN7','KNN11','KNN many neighbors')


#EXPORT CSV FILE WITH TOP 20 MOST SIGNIFICANT VARIABLES AND RESULTS
#export_csv = df2.to_csv (r'D:\Archivos\Machine Learning Centraal\Futbol ML\KNN20.csv',\
export_csv = df2.to_csv (r'KNN20.csv',\
                        index = None, header=True,columns=['Dirigidos','G_PPORJ2','DiffValJug',\
                                         'ValorLocal', 'G_PPORJ3','DiffValor','PosLoc',\
                                         'ValorVisitante','Distancia','G_PUNTOS3',\
                                         'G_PUNTOS2','DIFG5_AD','DT_PP','DT_PE',\
                                         'LV_PPORJ2','DT_PG','GC4_AD','DiffEdadProm', \
                                         'DIFG4_AD','G_EFECT3','Resultado']) 
