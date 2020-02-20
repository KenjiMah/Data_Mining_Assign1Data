# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:56:22 2020

@author: Kenji_Mah
"""

import pandas as pd

import numpy as np 
from os import listdir
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

GroundTruthUsers = listdir("groundTruth")
MyoDataUsers = listdir("MyoData")

A = set(GroundTruthUsers)
B = set(MyoDataUsers) 
print (A.difference(B)) 
print (B.difference(A)) 

intersection = list(A.intersection(B))
intersection.sort()
eatingAggregationOfFrames = []
nonEatingAggregationOfFrames = []
userDataMatrix = pd.DataFrame()
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
    path = "{}\{}\{}".format(folder,userFolder1,utensil)
    filename = listdir(path)[0][:-4]
    path = "{}\{}\{}\{}".format("MyoData",userFolder2,utensil,filename)
    
    data = pd.read_csv(path + "_{}.txt".format(dataType), sep=",", header=None, names = columns)
    
    
    videoInfo = pd.read_csv(path + "_video_info.csv", sep=",", header=None, usecols=[0,1], names=['rate', 'lastFrame'])
    lastFrame = videoInfo.loc[0]['lastFrame']
    path = "{}\{}\{}\{}".format(folder,userFolder1,utensil,filename)
    groundTruth = pd.read_csv(path + ".txt", sep=",", header=None, usecols=[0,1])
    #convert the frequency of the ground truth to match emg freq
    
    if dataType == 'EMG':
        groundTruth = groundTruth.apply(lambda x: round((x - lastFrame)* 100 / videoInfo.loc[0]['rate']))
    elif dataType == 'IMU':
        groundTruth = groundTruth.apply(lambda x: round((x - lastFrame) * 50 / videoInfo.loc[0]['rate']))
    else:
        exit("wrong data source")
    #keep track of the span of eating frames
    for index, row in groundTruth.iterrows():
        global eatingAggregationOfFrames
        if row[1]- row[0] < 0:
            print (path)
            continue
        eatingAggregationOfFrames.append(row[1]- row[0])

    return LabelData(groundTruth, data)

def display_ffts(dataSet, attribute):
    if attribute.startswith('Accelerometer') or attribute.startswith('Orientation'):
        plt.figure(figsize=(20,10))
    elif attribute.startswith('EMG'):
        pass
    elif attribute.startswith('Gyroscope'):
        plt.figure(figsize=(20,10))
    else:
        plt.figure()
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
        
        plt.plot(X, 2.0*np.abs(sig_fft[:N])/N, linewidth=1)
        plt.xlabel('Frequency ($Hz$)')
        plt.ylabel('Amplitude ($Unit$)')
        plt.title('fft of ' + attribute)
        if index == 0:
            freq1 = 2.0*np.abs(sig_fft[:N])/N
        else:
            freq2 = 2.0*np.abs(sig_fft[:N])/N
        index += 1
    if attribute.startswith('Accelerometer') or attribute.startswith('Orientation'):
        plt.axis([-.000000025,.0000025,0, 3])

    elif attribute.startswith('EMG'):
        pass
    elif attribute.startswith('Gyroscope'):
        plt.axis([-.0000002,.00002,0, 4])
    else:
        pass
    plt.legend(['eating', 'non-eating'])

    max1 = freq1.argsort()[-5:][::-1]
    max2 = freq2.argsort()[-5:][::-1]
    maxFreqs1 = []
    maxFreqs2 = []
    for i in max1:
        maxFreqs1.append(X[i])
    for i in max2:
        maxFreqs2.append(X[i])
    return maxFreqs1.sort(), maxFreqs2.sort()

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



def plot_frequencies(attributes, fftTop, FFTMatrixEating, FFTMatrixNonEating):
    plt.figure(figsize=(18,7))
    maxdf = max( FFTMatrixEating[attributes].values.max(), FFTMatrixNonEating[attributes].values.max())
    lineNames = []
    for i in range(len(FFTMatrixEating)):
        lineNames.append('topFreq' + str(i))
    for i in range(fftTop):
        plt.plot(attributes, FFTMatrixEating.loc[i][attributes], linewidth=2)
        plt.legend(lineNames)
        plt.xlabel('Feature')
        plt.ylabel('Frequency ($Hz$)')
        plt.title('Top FFT freqs of eating')
        plt.yticks(np.linspace(0,maxdf,10))
    #plt.plot(attributes, FFTMatrixNonEating.loc[i][attributes], linewidth=2, linestyle = '-')
    plt.figure(figsize=(18,7))
    for i in range(fftTop):
        plt.plot(attributes, FFTMatrixNonEating.loc[i][attributes], linewidth=2, linestyle='-')
        plt.legend(lineNames)
        plt.xlabel('Feature')
        plt.ylabel('Frequency ($Hz$)')
        plt.title('Top FFT freqs of Non-Eating')
        plt.yticks(np.linspace(0,maxdf,10))
    return
def get_PCAInput_for_sample(eatingMatrix, nonEatingMatrix, plot = False):   
    FFTMatrixEating = pd.DataFrame(columns = eatingMatrix.columns[1:-1], data = np.zeros((5,len(eatingMatrix.columns[1:-1]))))
    FFTMatrixNonEating = pd.DataFrame(columns = eatingMatrix.columns[1:-1], data = np.zeros((5,len(eatingMatrix.columns[1:-1]))))
    for col in eatingMatrix.columns[1:-1]:
        maxFreqs1, maxFreqs2 = get_top_N_FFTs([eatingMatrix[:], nonEatingMatrix[:len(eatingMatrix)]], col, fftTop)
        for i in range(fftTop):
            FFTMatrixEating.loc[i,col] = maxFreqs1[i]
            FFTMatrixNonEating.loc[i,col] = maxFreqs2[i]
    
    statisticsEating = eatingMatrix.describe().T
    statisticsNonEating = nonEatingMatrix.describe().T
    statisticsEating.reset_index()
    statisticsNonEating.reset_index()
    def plot_stat(stat):
        plt.figure(figsize=(6,4))
        plt.plot(statisticsEating.iloc[1:][stat])
        plt.plot(statisticsNonEating.iloc[1:][stat])
        plt.xticks(eatingMatrix.columns[1:-1], rotation = 90)
        plt.legend(['eating', 'non-eating'])
        plt.xlabel("attribute")
        plt.ylabel(stat)
        return
    if plot:
        plot_frequencies(eatingMatrix.columns[1:-1], fftTop, FFTMatrixEating, FFTMatrixNonEating)
        plot_stat('mean')
        plot_stat('std')
        plot_stat('max')
        plot_stat('min')
        
    setupCols = []
    for col in FFTMatrixEating.columns:
        for i in range(fftTop):
            setupCols.append(col + ' topFFT' + str(i + 1))
            
        setupCols.append(col + ' mean')
        setupCols.append(col + ' std')
        setupCols.append(col + ' max')
        setupCols.append(col + ' min')
        
        
    phase2Matrix = pd.DataFrame(columns = setupCols, data = np.zeros((2, len(setupCols))))
    
    for col in FFTMatrixEating.columns:
        for i in range(fftTop):
            phase2Matrix.loc[0][col + ' topFFT' + str(i + 1)] = FFTMatrixEating.loc[i][col]
            phase2Matrix.loc[1][col + ' topFFT' + str(i + 1)] = FFTMatrixNonEating.loc[i][col]
        phase2Matrix.loc[0][col + ' mean'] = statisticsEating.loc[col]['mean']
        phase2Matrix.loc[0][col + ' std'] = statisticsEating.loc[col]['std']
        phase2Matrix.loc[0][col + ' min'] = statisticsEating.loc[col]['min']
        phase2Matrix.loc[0][col + ' max'] = statisticsEating.loc[col]['max']
        phase2Matrix.loc[1][col + ' mean'] = statisticsEating.loc[col]['mean']
        phase2Matrix.loc[1][col + ' std'] = statisticsEating.loc[col]['std']
        phase2Matrix.loc[1][col + ' min'] = statisticsEating.loc[col]['min']
        phase2Matrix.loc[1][col + ' max'] = statisticsEating.loc[col]['max']
        
    return phase2Matrix

def get_Phase2_Matrix(dataType, fftTop, rowAgg):
    
    global eatingAggregationOfFrames
    eatingAggregationOfFrames = []
    eatingMatrix = pd.DataFrame()
    nonEatingMatrix = pd.DataFrame()
    forkEMG = GetLabeledDataMatrix('groundTruth', 'user9', 'user09', 'fork', dataType)
    spoonEMG = GetLabeledDataMatrix('groundTruth', 'user9', 'user09', 'spoon', dataType)
    global userDataMatrix 
    userDataMatrix = pd.DataFrame()
    userDataMatrix = forkEMG.append(spoonEMG, ignore_index=True)
    #split matrix based on eating and not-eating
    eatingMatrix = eatingMatrix.append(userDataMatrix[userDataMatrix['label'] == 'eating'], ignore_index=True)
    nonEatingMatrix = nonEatingMatrix.append(userDataMatrix[userDataMatrix['label'] == 'not-eating'], ignore_index=True)
    
    for user in intersection:
        print(user)
        forkEMG = GetLabeledDataMatrix('groundTruth', user, user, 'fork', dataType)
        spoonEMG = GetLabeledDataMatrix('groundTruth', user, user, 'spoon', dataType)
        userDataMatrix = forkEMG.append(spoonEMG, ignore_index=True)
        eatingMatrix = eatingMatrix.append(userDataMatrix[userDataMatrix['label'] == 'eating'], ignore_index=True)
        nonEatingMatrix = nonEatingMatrix.append(userDataMatrix[userDataMatrix['label'] == 'not-eating'], ignore_index=True)
    if rowAgg == -1:
        rowAgg = len(eatingMatrix)
    # for col in eatingMatrix.columns[1:-1]:
    #     plt.figure(figsize=(40, 20))
    #     ax = eatingMatrix[:].plot.line( y = col , use_index=True, linewidth=.1)
    #     nonEatingMatrix[:len(eatingMatrix)].plot.line( y = col, use_index=True, color = 'orange', ax = ax, linewidth=.1)
    #     ax.legend(['eating', 'non-eating'])
    #     ax.set_xlabel("frames")
    #     ax.set_ylabel(col) 
    eatingAggregationOfFrames = pd.DataFrame(eatingAggregationOfFrames, columns = ['span of eatting (frames)'])
    eatingAggregationOfFrames.hist()
    finalPhase2Matrix = pd.DataFrame()
    phase2Matrix = pd.DataFrame()
    for i in range(int(len(eatingMatrix)/rowAgg)):
        if i % 100 == 0:     
            print(i)
        phase2Matrix = get_PCAInput_for_sample(eatingMatrix[i * rowAgg: (i + 1) * rowAgg], nonEatingMatrix[i * rowAgg: (i + 1) * rowAgg])
        phase2Matrix = pd.concat([phase2Matrix, pd.DataFrame(columns = ['target'],data = ['eating', 'non-eating'])], axis = 1)
        finalPhase2Matrix = finalPhase2Matrix.append(phase2Matrix, ignore_index=True)
    return finalPhase2Matrix, eatingMatrix, nonEatingMatrix


################################################################ main ########################################################
fftTop = 5
rowAgg = 100
phase2MatrixEMG, phase1EatingMatrixEMG, phase1NonEatingMatrixEMG = get_Phase2_Matrix('EMG', fftTop, rowAgg * 2)
phase2MatrixIMU, phase1EatingMatrixIMU, phase1NonEatingMatrixIMU = get_Phase2_Matrix('IMU', fftTop, rowAgg)

# testplot = userDataMatrix[['TimeStamp','Gyroscope_Y', 'label' ]]
# plt.figure(figsize=(100, 20))
# fig, ax = plt.subplots(figsize=(100, 20))
# colors = {'eating':'red', 'not-eating':'blue'}
# ax.scatter(testplot['TimeStamp'], testplot['Gyroscope_Y'], c=testplot['label'].apply(lambda x: colors[x]), s=.1)
# StandardScaler().fit_transform(data)
phase1EatingMatrixIMU[phase1EatingMatrixIMU.columns[1:-1]] = StandardScaler().fit_transform(phase1EatingMatrixIMU[phase1EatingMatrixIMU.columns[1:-1]])
phase1EatingMatrixEMG[phase1EatingMatrixEMG.columns[1:-1]] = StandardScaler().fit_transform(phase1EatingMatrixEMG[phase1EatingMatrixEMG.columns[1:-1]])
get_PCAInput_for_sample(phase1EatingMatrixIMU[:], phase1NonEatingMatrixIMU[:], plot = True)
get_PCAInput_for_sample(phase1EatingMatrixEMG[:], phase1NonEatingMatrixEMG[:], plot = True)


################## do PCA #############################################

PCAInput = pd.concat([phase2MatrixIMU.iloc[:len(phase2MatrixEMG), : len(phase2MatrixIMU.columns) - 1], phase2MatrixEMG], axis=1, sort=False)
y = PCAInput.loc[:]['target'].values
x = StandardScaler().fit_transform(PCAInput.iloc[:, :(len(PCAInput.columns) - 1)])
from sklearn.decomposition import PCA
pca = PCA()
principleComponentsFeatureMatrix = pca.fit_transform(x)

######### prepare top 5 PC tables #################################
top5eigenvectors = pd.DataFrame(np.square(pca.components_)[:5], columns = PCAInput.columns[:-1])
top5eigenvectors = top5eigenvectors.T.rename(columns={0:'PC1',1:'PC2',2:'PC3',3:'PC4',4:'PC5'})
PC1Vecs = top5eigenvectors['PC1'].sort_values(ascending=False)
PC2Vecs = top5eigenvectors['PC2'].sort_values(ascending=False)
PC3Vecs = top5eigenvectors['PC3'].sort_values(ascending=False)
PC4Vecs = top5eigenvectors['PC4'].sort_values(ascending=False)
PC5Vecs = top5eigenvectors['PC5'].sort_values(ascending=False)
####### plot 2 pca  ####################################
principleDf = pd.DataFrame(data = principleComponentsFeatureMatrix[:, 0:2], columns = ['principle component 1', 'principle component 2'])
principleDf.head()
finalDF = pd.concat([principleDf, pd.DataFrame(columns = ['target'],data = y)], axis = 1)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('top 2 components of PCA', fontsize = 20)
targets = ['eating', 'non-eating']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indiciesToKeep = finalDF['target'] == target
    ax.scatter(finalDF.loc[indiciesToKeep, 'principle component 1'], finalDF.loc[indiciesToKeep, 'principle component 2'], c = color, s = 1)
ax.legend(targets)
ax.grid()



##### plot spider plot ##############################
spiderPlot = pca.explained_variance_ratio_

spiderPlot2 = []
for i in range(len(spiderPlot)):
    spiderPlot2.append(np.sum(spiderPlot[:i]))
plt.figure(figsize=(8,8))
plt.plot(np.linspace(0, 100, num=100),  100 * spiderPlot[:100])
plt.plot(np.linspace(0, 100, num=100),  np.multiply(spiderPlot2[:100], 100))
plt.yticks(np.arange(0, 100, step=10), rotation = 90)
plt.legend(['PCA components', 'cumulative percent of variation'])
plt.xlabel("Principal Components")
plt.ylabel('Variance Explained %')


