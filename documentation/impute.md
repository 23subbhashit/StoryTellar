
# Impute

```python3
storyscience.impute(array, method="mean")
```
Imputes values in an array containg nan values,based on two methods: mean and mode

**Arguments**

- **array** : An Array, containing nan values which has to be imputed.
- **method** : A String , specifying the type of method used for imputing.types={"mean","mode"}.default="mean"

**Example**

```
a=pd.read_csv("/kaggle/input/retaildataset/sales data-set.csv")
b=ss.impute(a['Dept'])
```

**Dataset**

<a href="https://www.kaggle.com/manjeetsingh/retaildataset" target="_blank">Retail Data Analytics</a>



