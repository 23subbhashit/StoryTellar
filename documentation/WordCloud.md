# WordCloud

```python3
storyscience.wordarraycloud(arr_words, width=800, height=400)
```
Utility for plotting a wordcloud from a given array of words.

**Arguments**

- **data** : An Array of words.
- **width** : An Integer specifying width of the cloud.default=800
- **height** : An Integer specifying height of the cloud.default=400


**Example**

```
data= pd.read_csv('../input/pandas-bokeh/long_data_.csv')
storyscience.wordarraycloud(data['States'])
```
**Dataset**

<a href="https://www.kaggle.com/smart1004/pandas-bokeh" target="_blank">Pandas Bokeh</a>





