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
        if len(maxFreqs1) < fftTop:
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
    
def get_metrics(user, y_test, y_pred):
    table = []
    table.append(user)
    table.append(metrics.accuracy_score(y_test, y_pred))
    table.append(metrics.precision_score(y_test, y_pred))
    table.append(metrics.recall_score(y_test, y_pred))
    table.append(metrics.f1_score(y_test, y_pred))
    return table 

def get_tree_metrics(data, user):
    if user != 'allusers':
        userData = data.groupby('user').get_group(user)
    else:
        userData = data
    userData = userData.drop('user', axis = 1)
    X = userData.drop('label', axis = 1)
    y = userData['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)
    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    pca = PCA(.95)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test) 
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    return get_metrics(user, y_test, y_pred)

def get_svm_metrics(data, user):
    if user != 'allusers':
        userData = data.groupby('user').get_group(user)
    else:
        userData = data
    userData = userData.drop('user', axis = 1)
    X = userData.drop('label', axis = 1)
    y = userData['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)
    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    pca = PCA(.95)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test) 
    
    svclassifier = SVC(kernel='poly', degree=3)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    return get_metrics(user, y_test, y_pred)

def get_NN_metrics(data, user):
    if user != 'allusers':
        userData = data.groupby('user').get_group(user)
    else:
        userData = data
    userData = userData.drop('user', axis = 1)
    X = userData.drop('label', axis = 1)
    y = userData['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)
    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    pca = PCA(.95)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test) 
    
    

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 5), random_state=1, max_iter = 2000)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return get_metrics(user, y_test, y_pred)
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


inputData = wholeIMUMatrix

#########################   do decision tree   ################################
decisionTreeMetricsTable = [] 
decisionTreeMetricsTable.append(get_tree_metrics(inputData, 'user9'))
for user in intersection:
    decisionTreeMetricsTable.append(get_tree_metrics(inputData, user))
decisionTreeMetricsTable = pd.DataFrame(data = decisionTreeMetricsTable, columns= ['user', 'accuracy', 'precision', 'recall', 'f1'])

#########################   do SVM  ################################


SVMMetricsTable = [] 
SVMMetricsTable.append(get_svm_metrics(inputData, 'user9'))
for user in intersection:
    SVMMetricsTable.append(get_svm_metrics(inputData, user))
SVMMetricsTable = pd.DataFrame(data = SVMMetricsTable, columns= ['user', 'accuracy', 'precision', 'recall', 'f1'])

########################## do neural network  ######################################

NNMetricsTable = [] 
NNMetricsTable.append(get_NN_metrics(inputData, 'user9'))
for user in intersection:
    NNMetricsTable.append(get_NN_metrics(inputData, user))
NNMetricsTable = pd.DataFrame(data = NNMetricsTable, columns= ['user', 'accuracy', 'precision', 'recall', 'f1'])




allUsersMetricsTable = []
allUsersMetricsTable.append(get_tree_metrics(inputData, 'allusers'))
allUsersMetricsTable.append(get_svm_metrics(inputData, 'allusers'))
allUsersMetricsTable.append(get_NN_metrics(inputData, 'allusers'))
allUsersMetricsTable = pd.DataFrame(data = allUsersMetricsTable, columns= ['method', 'accuracy', 'precision', 'recall', 'f1'])
allUsersMetricsTable['method'] = ['Decision Tree', 'Support Vector Machine', 'Neural Network']


# test1 = user_extracted_features('IMU', fftTop, ['user18', 'user25'], 100)
# test2 = user_extracted_features('EMG', fftTop, ['user18', 'user25'], 100)

# test1['label'].count
# users1 = wholeIMUMatrix.groupby('user')['label'].agg('count')

# users2 = wholeEMGMatrix.groupby('user')['label'].agg('count')


# user18IMU = wholeIMUMatrix.groupby('user').get_group('user18')
# user18EMG =wholeEMGMatrix.groupby('user').get_group('user18')




 

























