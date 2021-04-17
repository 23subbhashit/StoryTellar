# Numerical vs Numerical Plot

```python3
storyscience.numvsnumplot(arr1,arr2,stacked=True,scattercolor='c',linecolor='r')
```
Utility for plotting numerical vs numerical data using various plotting methods.

**Arguments**

- **arr1** : An Array , containing categorical values 
- **arr2** : An Array , containing categorical values

- **stacked** : A Boolean value, True for stacked plot.
- **scattercolor** : A String, denoting color type
- **linecolor** : A String, denoting color type

**Example**

```
a=pd.read_csv("/kaggle/input/retaildataset/Features data set.csv")
ss.catvscatplot(a['Dept'][:100000],a['Store'][:100000])
```
**Dataset**

<a href="https://www.kaggle.com/manjeetsingh/retaildataset" target="_blank">Retail Data Analytics</a>






