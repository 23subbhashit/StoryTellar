# Categorical vs Numerical Plot

```python3
storyscience.catvsnumericalplot(data,catcol,numcol,stacked=True,swarmcolor='c',violincolor='r',kdecolor='y',scattercolor='b',linecolor='g')
```
Utility for plotting categorical vs numerical data using various plotting methods.

**Arguments**

- **data** : The whole dataset only.
- **catcol** : A String, denoting categorical column name
- **numcol** : A String, denoting numerical column name
- **stacked** : A Boolean value, True for stacked plot
- **swarmcolor** : A String, denoting color type
- **violincolor** : A String, denoting color type
- **kdecolor** : A String, denoting color type
- **scattercolor** : A String, denoting color type
- **linecolor** : A String, denoting color type


**Example**

```
a=pd.read_csv("/kaggle/input/retaildataset/Features data set.csv")
ss.catvsnumericalplot(a[:1000],'Dept',"Weekly_Sales")
```
**Dataset**

<a href="https://www.kaggle.com/manjeetsingh/retaildataset" target="_blank">Retail Data Analytics</a>





