
# Describe

```python3
storyscience.zscore(data,threshold=1)
```
Computes ZScore and gives as output a list of outliers in the array

**Arguments**

- **data** : An Array containg numerics values.
- **threshold** : A float or int, denoting threshold value above which the number will be identified as an outlier.default=1

**Example**

```
a=pd.read_csv("/kaggle/input/retaildataset/sales data-set.csv")
b=zscore(a['Weekly_Sales'],15981.258123467243) 
```
**Dataset**

<a href="https://www.kaggle.com/manjeetsingh/retaildataset" target="_blank">Retail Data Analytics</a>






