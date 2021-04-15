
<p align="center">
    <img src="https://github.com/23subbhashit/StoryTellar/blob/master/Welcome%20to%20Vectr%20(2).svg" width="400" height="400"><br/>
    Simple yet flexible Data Analytics tool for Data Scientists.
</p>
<p align="center"">
  <a href="https://github.com/23subbhashit/StoryTellar/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/23subbhashit/StoryTellar"></a>
  <a href="https://github.com/23subbhashit/StoryTellar/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/23subbhashit/StoryTellar"></a>
  <a href="https://github.com/23subbhashit/StoryTellar/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/23subbhashit/StoryTellar"></a>
  <a href="https://github.com/23subbhashit/StoryTellar/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/23subbhashit/StoryTellar"></a>
  <a href="https://badge.fury.io/py/storyscience"><img src="https://badge.fury.io/py/storyscience.svg" alt="PyPI version" height="18"></a>
</p>

## What is it?

**storyscience** is a Python package that provides flexible, and easy to use functions for various anayltical operations.
It aims to provide support for bussiness analytics and nlp purposes.Currently , it's in the developing stage and we would 
like to improve it further and extend its support for various platforms.

## Main Features
Here are just a few of the things that it can provide:

  - Easy handling of missing data.
  - Provides summary of data and error scores of models for regression and classification.
  - Provides suggestive methods for  imputing and deleting rows and columns.
  - Support for map visuals,using plotly.
  - Provides statistical methods like zscore and iqr(interquartile range)
  - Support for categorical and numerical data plotting, using matplotlib and seaborn.

## Documentation
The  documentation for this can be found here : [Getting Started](https://pandas.pydata.org/pandas-docs/stable/install.html#installing-from-source).

## Installation

```
pip install storyscience
```

## Build

```
python setup.py sdist bdist_wheel
```

## PYPI

```
twine upload dist/*
```
