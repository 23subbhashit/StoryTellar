from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Subbhashit():
    return ('Hi Vro')

def Shree():
    return ("HI SHREE")

def Shivang():
    return "HI GUJJU"


def Count(x):
    dictionary = dict()
    array = list(x)
    countArray = dict(Counter(array))
    return countArray


def Impute(array,method='mean'):
    arr = list(array)
    pos = []
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            pos.append(i)
    for i in pos:
        arr.remove(arr[i])
    if method=='mean':
        for i in pos:
            key = int(sum(arr)/len(arr))
            arr.insert(i,key)
    elif method=='mode':
        for i in pos:
            dictionary = dict(Counter(arr).most_common(1))
            key = int(list(dictionary.keys())[0])
            arr.insert(i,key)
    return arr
    
def ZScore(data,threshold=1):
    threshold = 3
    outliers = []
    arr = list(data)
    mean = np.mean(arr)
    std = np.std(arr)
    for i in arr:
        z = (i-mean)/std
        if z > threshold:
            outliers.append(i)
    return outliers 

def SinglePlot(arr):
    fig, ax =plt.subplots(2,2)
    fig.set_size_inches(12.7, 10.27)
    
    plt.subplot(2,2,1)
    arr.value_counts().tail().plot(kind='pie',figsize=(15,10))
    
    sns.distplot(arr,ax=ax[0,1])
    
    plt.subplot(2, 2,3)
    arr.value_counts().tail().plot(kind='bar',color=['c','y','r'],figsize=(15,10))
    
    sns.boxplot(arr,ax=ax[1,1])
    
    
    fig.show()
