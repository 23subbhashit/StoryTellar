
# Suggest Similar

```python3
storyscience.suggest_similar(df, unique_id, col)
```
Suggests which columns should be droped in a dataframe using a threshold value.

**Arguments**

- **data** : An Array , containing categorical/numerical values 
- **unique_id** :  A String , denoting column name on which similarity algorithm has to be applied.
- **col** : A string  , denoting column name containing textual data using which similarity has to be applied.

**Example**

```
df = pd.read_csv("C:/mov.csv")
ss.suggest_similar(df, "title", "plot")
```
**Dataset**

<a href="https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection" target="_blank">News Headlines Dataset For Sarcasm Detection</a>







