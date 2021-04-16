# Suggest Category of Column

```python3
storyscience.suggest_cats(data, th=40)
```
Suggests which columns should be considered as categorical and numerical in a dataframe using a threshold value.

**Arguments**

- **data** : The whole dataset only.
- **th** :  An Int , denoting threshold which is used for comparision.default=40.

**Example**

```
a=pd.read_csv("/kaggle/input/retaildataset/Features data set.csv")
ss.suggest_cats(data, th=40)
```

**Dataset**

<a href="https://www.kaggle.com/manjeetsingh/retaildataset" target="_blank">Retail Data Analytics</a>
