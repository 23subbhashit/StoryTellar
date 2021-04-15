
# Describe

```python3
storyscience.describe(data)
```
Provides some basic intel about the dataset , like min, max , mean, median, mode,type ,length and standard deviation.

**Arguments**

- **data** : The whole dataset only.

**Example**

```
abc=pd.read_csv("/kaggle/input/retaildataset/Features data set.csv")
df=ss.describe(abc)
df.head(10)
```
**Dataset**

<a href="https://www.kaggle.com/manjeetsingh/retaildataset" target="_blank">Retail Data Analytics</a>





