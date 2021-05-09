# WordCloud

```python3
storyscience.time_series_plot(data, datefrom, dateto, text, col, figsize=(16, 9))
```
Utility for plotting a wordcloud from a given array of words.

**Arguments**

- **data** : The whole dataset only.
- **datefrom** : A String, denoting date value for getting data upto that specified date 
- **dateto** : A String, denoting date value for getting data after that specified date 
- **text** : A String , denoting title of plot
- **col** : Column on which the operation has to be performed
- **figsize** : Denoting the width and height of canvas for plotting.default=(16,9)


**Example**

```
data= pd.read_csv('../input/pandas-bokeh/long_data_.csv')
storyscience.wordarraycloud(data['States'])
```
**Dataset**

<a href="https://www.kaggle.com/smart1004/pandas-bokeh" target="_blank">Pandas Bokeh</a>





