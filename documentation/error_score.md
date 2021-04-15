
# ErrorScore

```python3
def error_score(yt,yp,typ='classification',beta=0.5,average='macro'):
```
Gives error scores for various actual and predicted values , applicable only for regression and classification purposes.

**Arguments**

- **yt** : An Array, containing country actual values.
- **yp** : An Array ,  containing predicted values.
- **typ** : A String , specifying for which operation scores has to be predicted.types={‘regression’, ‘classification’}.default='classification'
- **beta** : A Float,which determines the weight of recall in the combined score.default='0.5'
- **average** : A StringThis parameter is required for multiclass/multilabel targets.types={‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or None . default='macro'

**Example**

```
ss.error_score(yt,yp,'classification')
```
**Dataset**

<a href="https://www.kaggle.com/chaitanyahivlekar/fifa-19-player-dataset" target="_blank">FIFA 19 player dataset</a>



