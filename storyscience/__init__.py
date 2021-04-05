from collections import Counter
import numpy as np
import pandas as pd

def subbhashit():
    return ('Hi Vro')

def shree():
    return ("HI SHREE")

def shivang():
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