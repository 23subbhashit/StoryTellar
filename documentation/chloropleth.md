# Chloropleth

```python3
storyscience.chloropleth(data,title='', hue='')
```
Plots a map from the given data  with an appropriate title and hue name for countries.

**Arguments**

- **data** : An Array, containing country names.
- **title** : A String ,denoting tile for the map
- **hue** : A string , denoting colorbar title

**Example**

```
data=pd.read_csv("/kaggle/input/fifa-19-player-dataset/FIFA19.csv")
ss.chloropleth(data.Nationality,'Number of players from each country',"NO. of players")
```
**Dataset**

<a href="https://www.kaggle.com/chaitanyahivlekar/fifa-19-player-dataset" target="_blank">FIFA 19 player dataset</a>



