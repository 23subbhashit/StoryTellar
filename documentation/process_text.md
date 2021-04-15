# Text Processing

```python3
storyscience.process_text(x)
```
 For text cleaning and removal of unwanted characters.

**Arguments**

- **x** : A String, which has to be processed.


**Example**

```
data=pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json",lines=True)
data.headline=data.headline.apply(ss.process_text)
```
**Dataset**

<a href="https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection" target="_blank">News Headlines Dataset For Sarcasm Detection</a>



