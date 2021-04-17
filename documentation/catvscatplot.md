
# Categorical vs Categorical Plot

```python3
storyscience.catvscatplot(arr1,arr2,stacked=True)
```
Utility for plotting categorical vs categorical data using crosstabs.

**Arguments**

- **arr1** : An Array , containing categorical values 
- **arr2** : An Array , containing categorical values

- **stacked** : A Boolean value, True for stacked plot.

**Example**

```
a=pd.read_csv("/kaggle/input/retaildataset/Features data set.csv")
ss.catvscatplot(a['Dept'][:100000],a['Store'][:100000])
```
**Dataset**

<a href="https://www.kaggle.com/manjeetsingh/retaildataset" target="_blank">Retail Data Analytics</a>






