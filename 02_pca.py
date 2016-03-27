# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 11:13:19 2016

@author: 王晓捷 11521053
"""
import numpy as np
import matplotlib.pyplot as plt

topNfeat = 2
filepath='E:/WorkSpace_Python/PCA_dataset/optdigits.tra'
  
"""
get features of 64 dimension of the number 3 from the feature file
"""
def getData(filepath,number) :
    file = open(filepath)
    dataset = np.zeros((1,64))
    flag = -1
    for line in file:
        factor = line.split(',')
        if factor[64].find(number)!= -1:
            factor.pop()
            if flag == -1:
                dataset[0] = map(float,factor)
                flag = 0
            else :
                dataset = np.vstack((dataset,map(float,factor)))
    file.close()
    return dataset

"""
PCA process 
"""
def PCA(dataset):
    #calculate the mean value
    meanVals = np.mean(dataset,axis = 0)
    #minus value
    meanRemoved = dataset - meanVals
    #calculate covariance matrix
    covMat = np.cov(meanRemoved, rowvar = 0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[-1:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDMat = meanRemoved*redEigVects

    return lowDMat
    

"""
main function
"""
dataset = getData(filepath,'3')
lowDMat = PCA(dataset)
#draw the plot
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.array(lowDMat[:,0])
y = np.array(lowDMat[:,1])
ax.scatter(x,y,marker='o',s = 90,c='r')
plt.show()

