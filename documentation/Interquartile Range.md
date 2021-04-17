
# Interquartile Range(IQR)

```python3
storyscience.iqr(data, arg1=75, arg2=25)
```
Provides IQR value for an array.

**Arguments**

- **data** : An Array , containing numerical values 
- **arg1** : An Int , denoting value for upper quartile

- **arg2** : An Int, denoting value for lower quartile

**Example**

```
a=pd.read_csv("/kaggle/input/retaildataset/Features data set.csv")
rangeIQR = ss.iqr(a['Store'])
```
**Dataset**

<a href="https://www.kaggle.com/manjeetsingh/retaildataset" target="_blank">Retail Data Analytics</a>






