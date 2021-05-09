# Time-Series Plotting

```python3
storyscience.time_series_plot(data, datefrom, dateto, text, col, figsize=(16, 9))
```
Utility for plotting  time wise data.

**Arguments**

- **data** : The whole dataset only.
- **datefrom** : A String, denoting date value for getting data upto that specified date 
- **dateto** : A String, denoting date value for getting data after that specified date 
- **text** : A String , denoting title of plot
- **col** : Column on which the operation has to be performed
- **figsize** : Denoting the width and height of canvas for plotting.default=(16,9)


**Example**

```
data=pd.read_csv("../input/stock-price-predictions/Apple.csv",
                 index_col="Price Date",
                 parse_dates=["Price Date"])
storyscience.time_series_plot(data,'2016','2016','Apple stock price',"Modal Price (Rs./Quintal)")
```
**Dataset**

<a href="https://www.kaggle.com/subbhashit/stock-price-predictions" target="_blank">Stock Price Predictions</a>





