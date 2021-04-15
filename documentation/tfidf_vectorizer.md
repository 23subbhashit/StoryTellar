# TF-IDF

```python3
storyscience.tfidf_vectorizer(x,max_featues=1000,min_df=5,max_df=0.7)
```
Utility for plotting categorical vs categorical data using crosstabs.

**Arguments**

- **x** : An Array ,conataining text.
- **max_features** :An Int or Float, if not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.This parameter is ignored if vocabulary is not None.default=1000
- **min_df** : A Float or Int. When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float in range of [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.default=5

- **max_df** : A Float or Int,if not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

This parameter is ignored if vocabulary is not None.default=0.7

**Example**

```
data=pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json",lines=True)
df=ss.tfidf_vectorizer(data['headline'])
```
**Dataset**

<a href="https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection" target="_blank">News Headlines Dataset For Sarcasm Detection</a>







