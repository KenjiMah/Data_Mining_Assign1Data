# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:59:02 2020

@author: Kenji_Mah
"""

import pandas as pd

import numpy as np 
from os import listdir
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

def LabelData(groundTruth, data):
    #label the eating and non eating segments of data using the ground truth information
    data['label'] = 'not-eating'
    for index, row in groundTruth.iterrows():
        if (row[1]- row[0] < 0): 
            continue
        else:
            data[int(row[0]) : int(row[1] + 1)]['label'] = 'eating'
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

def get_top_N_FFTs(dataSet, attribute, top):
    freq1 = []
    freq2 = []
    index = 0
    for data in dataSet:
        #sample period is basically the row aggregation
        samplePeriod = 100
        sig_fft = np.fft.fft(data[attribute].values)
        N = int(len(sig_fft)/2)+1
        dt = samplePeriod
        fa = 1/dt
        X = np.linspace(0, fa/2, N, endpoint=True)
        
        if index == 0:
            freq1 = 2.0*np.abs(sig_fft[:N])/N
        else:
            freq2 = 2.0*np.abs(sig_fft[:N])/N
        index += 1
    
    max1 = freq1.argsort()[-(top):][::-1]
    max2 = freq2.argsort()[-(top):][::-1]
    maxFreqs1 = []
    maxFreqs2 = []
    for i in range(top):
        maxFreqs1.append(X[max1[i]])
    for i in range(top):
        maxFreqs2.append(X[max2[i]])
    maxFreqs1.sort()
    maxFreqs2.sort()
    return maxFreqs1, maxFreqs2

def extracted_features(dataType, fftTop, users):
    
    fullMatrix = pd.DataFrame()
    forkEMG = GetLabeledDataMatrix('groundTruth', 'user9', 'user09', 'fork', dataType)
    spoonEMG = GetLabeledDataMatrix('groundTruth', 'user9', 'user09', 'spoon', dataType)
    userDataMatrix = pd.DataFrame()
    userDataMatrix = forkEMG.append(spoonEMG)
    userDataMatrix['user'] = 'user9'
    fullMatrix.append(userDataMatrix)
    i = 0
    for user in users:
        if i > 3:
            break
        i += 1
        print(user)
        forkEMG = GetLabeledDataMatrix('groundTruth', user, user, 'fork', dataType)
        spoonEMG = GetLabeledDataMatrix('groundTruth', user, user, 'spoon', dataType)
        userDataMatrix = forkEMG.append(spoonEMG)
        userDataMatrix['user'] = user
        fullMatrix = fullMatrix.append(userDataMatrix)
        
    return fullMatrix
    
########################################    Main    #########################################

GroundTruthUsers = listdir("groundTruth")
MyoDataUsers = listdir("MyoData")

A = set(GroundTruthUsers)
B = set(MyoDataUsers) 
print (A.difference(B)) 
print (B.difference(A)) 

intersection = list(A.intersection(B))
fftTop = 5
wholeMatrix = pd.DataFrame()
wholeMatrix = extracted_features('IMU', fftTop, intersection)














































