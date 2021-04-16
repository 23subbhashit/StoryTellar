
# Suggest Fillers

```python3
storyscience.suggest_fillers(data, th=40)
```
Suggests which columns should be filled in a dataframe using a threshold value.

**Arguments**

- **data** : An Array , containing categorical/numerical values 
- **th** :  An Int , denoting threshold which is used for comparision.default=40.

**Example**

```
a=pd.read_csv("/kaggle/input/retaildataset/Features data set.csv")
ss.suggest_drops(a)
```
**Dataset**

<a href="https://www.kaggle.com/manjeetsingh/retaildataset" target="_blank">Retail Data Analytics</a>







