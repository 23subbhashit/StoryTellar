
# Cosine similarity for Dictionaries

```python3
storyscience.get_cosine_dict(vec1, vec2)
```
Provides cosine similarity for 2 dictionaries.

**Arguments**

- **vec1** : A Dictionary, containing words and their count as key-vale pairs.
- **vec2** : A Dictionary, containing words and their count as key-vale pairs.

**Example**

```
>>> l="i am dog"
>>> j="i am cat"
>>> l=ss.text_to_dict(l)
>>> j=ss.text_to_dict(j)
>>> l,j
(Counter({'i': 1, 'am': 1, 'dog': 1}), Counter({'i': 1, 'am': 1, 'cat': 1}))
>>> ss.get_cosine_dict(l,j)
0.6666666666666667
```





