# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:59:02 2020

@author: Kenji_Mah
"""

import pandas as pd
import time
import numpy as np 
from os import listdir
import os.path
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import random

def LabelData(groundTruth, data):
    #label the eating and non eating segments of data using the ground truth information
    data['label'] = 0
    for index, row in groundTruth.iterrows():
        if (row[1]- row[0] < 0): 
            continue
        else:
            data[int(row[0]) : int(row[1] + 1)]['label'] = 1
    return data

def GetLabeledDataMatrix(folder, userFolder1, userFolder2, utensil, dataType):
    if dataType == 'EMG':
        columns = ["TimeStamp", "EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"]
    elif dataType == 'IMU':
        columns = ["TimeStamp", "Orientation_X", "Orientation_Y", "Orientation_Z", "Orientation_W", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z",  "Gyroscope_X",  "Gyroscope_Y",  "Gyroscope_Z"]
    else:
        exit("wrong data source")
    path = "{}/{}/{}".format(folder,userFolder1,utensil)
    filename = listdir(path)[0][:-4]
    path = "{}/{}/{}/{}".format("MyoData",userFolder2,utensil,filename)
    
    data = pd.read_csv(path + "_{}.txt".format(dataType), sep=",", header=None, names = columns)
    data = data.astype(float)
    
    videoInfo = pd.read_csv(path + "_video_info.csv", sep=",", header=None, usecols=[0,1], names=['rate', 'lastFrame'])
    lastFrame = videoInfo.loc[0]['lastFrame']
    path = "{}/{}/{}/{}".format(folder,userFolder1,utensil,filename)
    groundTruth = pd.read_csv(path + ".txt", sep=",", header=None, usecols=[0,1])
    #convert the frequency of the ground truth to match emg freq
    
    if dataType == 'EMG':
        groundTruth = groundTruth.apply(lambda x: round((x - lastFrame)* 100 / videoInfo.loc[0]['rate']))
    elif dataType == 'IMU':
        groundTruth = groundTruth.apply(lambda x: round((x - lastFrame) * 50 / videoInfo.loc[0]['rate']))
    else:
        exit("wrong data source")

    return LabelData(groundTruth, data)

def sample_feature_extraction(sampleGesture, fftTop, samplePeriod):
    FFTFeatures = pd.DataFrame(columns = sampleGesture.columns[1:-1], data = np.zeros((fftTop,len(sampleGesture.columns[1:-1]))))
    freq1 = []
    for col in sampleGesture.columns[1:-1]:
        #sample period is basically the row aggregation
        sig_fft = np.fft.fft(sampleGesture[col].values)
        N = int(len(sig_fft)/2)+1
        dt = samplePeriod
        fa = 1/dt
        X = np.linspace(0, fa/2, N, endpoint=True)
        freq1 = 2.0*np.abs(sig_fft[:N])/N
        
        max1 = freq1.argsort()[-(fftTop):][::-1]
        maxFreqs1 = []
        if len(max1) < fftTop:
            maxFreqs1 = np.zeros(fftTop)
        else:
            for i in range(fftTop):
                maxFreqs1.append(X[max1[i]])
        maxFreqs1.sort()
        
        for i in range(fftTop):
            FFTFeatures.loc[i,col] = maxFreqs1[i]
    statistics = sampleGesture.describe().T
    statistics.reset_index()
    setupCols = []
    for col in FFTFeatures.columns:
        for i in range(fftTop):
            setupCols.append(col + ' topFFT' + str(i + 1))
        setupCols.append(col + ' mean')
        setupCols.append(col + ' std')
        setupCols.append(col + ' max')
        setupCols.append(col + ' min')
    
    newFeatures = pd.DataFrame(columns = setupCols, data = np.zeros((1, len(setupCols))))
    
    for col in FFTFeatures.columns:
        for i in range(fftTop):
            newFeatures.loc[0][col + ' topFFT' + str(i + 1)] = FFTFeatures.loc[i][col]
        newFeatures.loc[0][col + ' mean'] = statistics.loc[col]['mean']
        newFeatures.loc[0][col + ' std'] = statistics.loc[col]['std']
        newFeatures.loc[0][col + ' min'] = statistics.loc[col]['min']
        newFeatures.loc[0][col + ' max'] = statistics.loc[col]['max']
    newFeatures['label'] = sampleGesture.loc[0]['label']
    return newFeatures

def feature_extraction(data, n):
    PCAInput1 = pd.DataFrame()
    lastLabel = data.loc[0]['label']
    sample = pd.DataFrame(columns = data.columns)
    j = 0
    for i in range(len(data)):
        if data.loc[i]['label'] == lastLabel or i == len(data) - 1:
            sample.loc[j] = data.loc[i]
            j += 1
        else:
            PCAInput1 = PCAInput1.append(sample_feature_extraction(sample, fftTop, n))
            sample = pd.DataFrame(columns = data.columns)
            sample.loc[0] = data.loc[i]
            lastLabel = data.loc[i]['label']
            j = 1
    return PCAInput1

def user_extracted_features(dataType, fftTop, users, n):
    fullMatrix = pd.DataFrame()
    fork = pd.DataFrame(GetLabeledDataMatrix('groundTruth', 'user9', 'user09', 'fork', dataType))
    spoon = GetLabeledDataMatrix('groundTruth', 'user9', 'user09', 'spoon', dataType)
    fork = feature_extraction(fork, n)
    spoon = feature_extraction(spoon, n)
    userDataMatrix = pd.DataFrame()
    userDataMatrix = fork.append(spoon, ignore_index =True)
    userDataMatrix['user'] = 'user9'
    fullMatrix = fullMatrix.append(userDataMatrix, ignore_index =True)
    for user in users:
        print(user)
        fork = GetLabeledDataMatrix('groundTruth', user, user, 'fork', dataType)
        spoon = GetLabeledDataMatrix('groundTruth', user, user, 'spoon', dataType)
        fork = feature_extraction(fork, n)
        spoon = feature_extraction(spoon, n)
        userDataMatrix = fork.append(spoon, ignore_index =True)
        userDataMatrix['user'] = user
        fullMatrix = fullMatrix.append(userDataMatrix, ignore_index =True)
    fullMatrix.reset_index()
    return fullMatrix
    
def get_metrics(y_test, y_pred):
    table = []
    table.append(metrics.accuracy_score(y_test, y_pred))
    table.append(metrics.precision_score(y_test, y_pred))
    table.append(metrics.recall_score(y_test, y_pred))
    table.append(metrics.f1_score(y_test, y_pred))
    return table 

def get_tree_metrics(X_train, X_test, y_train, y_test):    
    clf = tree.DecisionTreeClassifier(max_depth = 2)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    plt.figure(figsize = (100,100))
    tree.plot_tree(clf.fit(X_train, y_train)) 
    return get_metrics(y_test, y_pred)

def get_svm_metrics(X_train, X_test, y_train, y_test):
    svclassifier = SVC(kernel='poly', degree=3)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    return get_metrics(y_test, y_pred)

def get_NN_metrics(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 5), random_state=1, max_iter = 2000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return get_metrics(y_test, y_pred)
    
    
########################################    Main    #########################################

GroundTruthUsers = listdir("groundTruth")
MyoDataUsers = listdir("MyoData")

A = set(GroundTruthUsers)
B = set(MyoDataUsers) 
print (A.difference(B)) 
print (B.difference(A)) 

intersection = list(A.intersection(B))
intersection.sort()
fftTop = 5
redo = False
############## get matricies for data ##################################
if os.path.isfile('IMUFeature.csv') and redo == False:
    wholeIMUMatrix = pd.read_csv('IMUFeature.csv' , sep=",")
    wholeIMUMatrix = wholeIMUMatrix.drop('Unnamed: 0', axis = 1)
else:
    start_time = time.time()
    wholeIMUMatrix = pd.DataFrame()
    wholeIMUMatrix = user_extracted_features('IMU', fftTop, intersection, 100)
    print("--- %s seconds ---" % (time.time() - start_time))
    wholeIMUMatrix.to_csv("IMUFeature.csv")

if os.path.isfile('EMGFeature.csv') and redo == False:
    wholeEMGMatrix = pd.read_csv('EMGFeature.csv' , sep=",")
    wholeEMGMatrix = wholeEMGMatrix.drop('Unnamed: 0', axis = 1)
else:
    start_time = time.time()
    wholeEMGMatrix = pd.DataFrame()
    wholeEMGMatrix = user_extracted_features('EMG', fftTop, intersection, 200)
    print("--- %s seconds ---" % (time.time() - start_time))
    wholeEMGMatrix.to_csv("EMGFeature.csv")
######### drop the 4 values because of missing IMU data ######################
wholeEMGMatrix = wholeEMGMatrix.drop(index = [1212,1213,1214])
wholeEMGMatrix = wholeEMGMatrix.reset_index(drop=True)
wholeEMGMatrix = wholeEMGMatrix.drop(index = [2238])
wholeEMGMatrix = wholeEMGMatrix.reset_index(drop=True)
for i in range(len(wholeEMGMatrix)):
    if wholeEMGMatrix.loc[i]['label'] != wholeIMUMatrix.loc[i]['label']:
        print('there is a missaligned input')
        
        
wholeMatrix = pd.concat([wholeIMUMatrix.drop(columns = ['label','user']), wholeEMGMatrix], axis = 1)

##########################   Phase 1 for each user ################################
inputData = wholeMatrix.groupby('user').get_group('user9')
inputData = inputData.drop('user', axis = 1)
X = inputData.drop('label', axis = 1)
y = inputData['label']
phase1X_train, phase1X_test, phase1Y_train, phase1Y_test = train_test_split(X, y, test_size=0.4, random_state=42)
for user in intersection:
    inputData = wholeMatrix.groupby('user').get_group(user)
    inputData = inputData.drop('user', axis = 1)
    X = inputData.drop('label', axis = 1)
    y = inputData['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    phase1X_train = phase1X_train.append(X_train)
    phase1Y_train = phase1Y_train.append(y_train)
    phase1X_test = phase1X_test.append(X_test)
    phase1Y_test = phase1Y_test.append(y_test)
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(phase1X_train)
# Apply transform to both the training set and the test set.
phase1X_train = scaler.transform(phase1X_train)
phase1X_test = scaler.transform(phase1X_test)
#do pca
pca = PCA(.95)
pca.fit(phase1X_train)
X_train = pca.transform(phase1X_train)
X_test = pca.transform(phase1X_test) 

phase1MetricsTable = []
phase1MetricsTable.append(get_tree_metrics(phase1X_train, phase1X_test, phase1Y_train, phase1Y_test))
phase1MetricsTable.append(get_svm_metrics(phase1X_train, phase1X_test, phase1Y_train, phase1Y_test))
phase1MetricsTable.append(get_NN_metrics(phase1X_train, phase1X_test, phase1Y_train, phase1Y_test))
phase1MetricsTable = pd.DataFrame(data = phase1MetricsTable, columns= ['accuracy', 'precision', 'recall', 'f1'])
phase1MetricsTable['method'] = ['Decision Tree', 'Support Vector Machine', 'Neural Network']

##########################   Phase 2 ################################
allUsers = intersection.copy()
allUsers.insert(0,'user9')
random.Random(42).shuffle(allUsers)
# split all users 60%, 40%
usersTrain = allUsers[:int(.6 * len(allUsers))].copy()
usersTest = allUsers[int(.6 * len(allUsers)):].copy()
inputData = wholeMatrix
phase2X_train = pd.DataFrame()
phase2X_test = pd.DataFrame()
test = wholeMatrix.groupby('user').get_group(user).loc[:, 'label']
for user in usersTrain:
    phase2X_train = phase2X_train.append(wholeMatrix.groupby('user').get_group(user).drop(['user'], axis = 1))
phase2Y_train = phase2X_train['label']
phase2X_train = phase2X_train.drop('label', axis = 1)
for user in usersTest:
    phase2X_test = phase2X_test.append(wholeMatrix.groupby('user').get_group(user).drop(['user'], axis = 1))
phase2Y_test = phase2X_test['label']
phase2X_test = phase2X_test.drop('label', axis = 1)

#  normalize data
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(phase2X_train)
# Apply transform to both the training set and the test set.
phase2X_train = scaler.transform(phase2X_train)
phase2X_test = scaler.transform(phase2X_test)
#do pca
pca = PCA(.95)
pca.fit(phase2X_train)
X_train = pca.transform(phase2X_train)
X_test = pca.transform(phase2X_test) 

phase2MetricsTable = []
phase2MetricsTable.append(get_tree_metrics(phase2X_train, phase2X_test, phase2Y_train, phase2Y_test))
phase2MetricsTable.append(get_svm_metrics(phase2X_train, phase2X_test, phase2Y_train, phase2Y_test))
phase2MetricsTable.append(get_NN_metrics(phase2X_train, phase2X_test, phase2Y_train, phase2Y_test))
phase2MetricsTable = pd.DataFrame(data = phase2MetricsTable, columns= ['accuracy', 'precision', 'recall', 'f1'])
phase2MetricsTable['method'] = ['Decision Tree', 'Support Vector Machine', 'Neural Network']






# ################ incorrect phase 2 on all data (keep just in case) #############################################
# inputData = wholeMatrix
# inputData = inputData.drop('user', axis = 1)
# X = inputData.drop('label', axis = 1)
# y = inputData['label']
# # split data
# phase2X_train, phase2X_test, phase2Y_train, phase2Y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# #  normalize data
# scaler = StandardScaler()
# # Fit on training set only.
# scaler.fit(phase2X_train)
# # Apply transform to both the training set and the test set.
# phase2X_train = scaler.transform(phase2X_train)
# phase2X_test = scaler.transform(phase2X_test)
# #do pca
# pca = PCA(.95)
# pca.fit(phase2X_train)
# X_train = pca.transform(phase2X_train)
# X_test = pca.transform(phase2X_test) 

# phase2MetricsTable = []
# phase2MetricsTable.append(get_tree_metrics(phase2X_train, phase2X_test, phase2Y_train, phase2Y_test))
# phase2MetricsTable.append(get_svm_metrics(phase2X_train, phase2X_test, phase2Y_train, phase2Y_test))
# phase2MetricsTable.append(get_NN_metrics(phase2X_train, phase2X_test, phase2Y_train, phase2Y_test))
# phase2MetricsTable = pd.DataFrame(data = phase2MetricsTable, columns= ['accuracy', 'precision', 'recall', 'f1'])
# phase2MetricsTable['method'] = ['Decision Tree', 'Support Vector Machine', 'Neural Network']














 

























