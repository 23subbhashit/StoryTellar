from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tabulate import tabulate as tb

warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import plotly.express as px
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_squared_log_error,
    make_scorer,
    median_absolute_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    fbeta_score,
)


def Subbhashit():
    return "Hi Vro"


def Shree():
    return "HI SHREE"


def Shivang():
    return "HI GUJJU"


def Count(x):
    array = list(x)
    countArray = dict(Counter(array))
    return countArray


def Impute(array, method="mean"):
    arr = list(array)
    pos = []
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            pos.append(i)
    for i in pos:
        arr.remove(arr[i])
    if method == "mean":
        for i in pos:
            key = int(sum(arr) / len(arr))
            arr.insert(i, key)
    elif method == "mode":
        for i in pos:
            dictionary = dict(Counter(arr).most_common(1))
            key = int(list(dictionary.keys())[0])
            arr.insert(i, key)
    return arr


def ZScore(data, threshold=1):
    threshold = 3
    outliers = []
    arr = list(data)
    mean = np.mean(arr)
    std = np.std(arr)
    for i in arr:
        z = (i - mean) / std
        if z > threshold:
            outliers.append(i)
    return outliers


def SinglePlot(arr):
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12.7, 10.27)

    plt.subplot(2, 2, 1)
    arr.value_counts().tail().plot(kind="pie", figsize=(15, 10))

    sns.distplot(arr, ax=ax[0, 1])

    plt.subplot(2, 2, 3)
    arr.value_counts().tail().plot(kind="bar", color=["c", "y", "r"], figsize=(15, 10))

    sns.boxplot(arr, ax=ax[1, 1])

    fig.show()


def IQR(data, arg1=75, arg2=25):
    q3, q1 = np.percentile(data, [arg1, arg2])
    iqr = q3 - q1
    return iqr


def Describe(data):
    l = list(data.columns)
    length = []
    mini = []
    maxi = []
    mean = []
    median = []
    mode = []
    typ = []
    std = []
    std = []
    types = ["float64", "int64"]
    for i in l:
        typ.append(data[i].dtype)
        length.append(len(data[i]))
        mini.append(min(data[i]))
        maxi.append(max(data[i]))
        if data[i].dtype in types:
            mean.append(data[i].mean())
            median.append(data[i].median())
            mode.append(data[i].mode()[0])
            std.append(np.std(data[i]))

        else:
            mean.append(np.nan)
            median.append(np.nan)
            mode.append(np.nan)
            std.append(np.nan)

    df = pd.DataFrame(
        [typ, length, mini, maxi, mean, median, mode, std],
        index=["Type", "Length", "Minimum", "Maximum", "Mean", "Median", "Mode", "STD"],
        columns=l,
    )
    return df


def Chloropleth(data, title="", hue=""):
    countries = data.value_counts()
    f = go.Figure(
        data=go.Choropleth(
            locations=countries.index,
            z=countries,
            locationmode="country names",
            colorscale=px.colors.sequential.Plasma,
            colorbar_title=str(hue),
        )
    )

    f.update_layout(
        title_text=str(title),
    )
    iplot(f)


def ErrorScore(yt, yp, typ="classification", beta=0.5, average="macro"):
    typ = typ.lower()
    r2_score1 = []
    mean_squared_error1 = []
    mean_squared_log_error1 = []
    median_absolute_error1 = []
    mean_absolute_error1 = []
    accuracy_score1 = []
    f1_score1 = []
    fbeta_score1 = []
    if typ == "regression":
        a = r2_score(yt, yp)
        b = mean_squared_error(yt, yp)
        c = mean_squared_log_error(yt, yp)
        d = median_absolute_error(yt, yp)
        e = mean_absolute_error(yt, yp)
        r2_score1.append(a)
        mean_squared_error1.append(b)
        mean_squared_log_error1.append(c)
        median_absolute_error1.append(d)
        mean_absolute_error1.append(e)
        df = pd.DataFrame(
            [
                r2_score1,
                mean_squared_error1,
                mean_squared_log_error1,
                median_absolute_error1,
                mean_absolute_error1,
            ],
            index=[
                "R2-SCORE",
                "MeanSquaredError",
                "MeanSquaredLogError",
                "MedianAbsoluteError",
                "MeanAbsoluteError",
            ],
            columns=["Score"],
        )
        return df
    elif typ == "classification":
        a = f1_score(yt, yp)
        b = accuracy_score(yt, yp)
        c = fbeta_score(yt, yp, beta=beta, average=average)
        f1_score1.append(a)
        accuracy_score1.append(b)
        fbeta_score1.append(c)
        df = pd.DataFrame(
            [accuracy_score1, f1_score1, fbeta_score1],
            index=["AUC", "F1-SCORE", "FBETA-SCORE"],
            columns=["Score"],
        )
        return df


def suggest_cats(data, th=40):
    dtb = []
    print(
        "Following columns might be considered to be changed as categorical\nTaking",
        th,
        "% as Threshold for uniqueness percentage determination\nLength of the dataset is:",
        len(data),
    )
    ln = len(data)

    for i in data.columns:
        unique_vals = data[i].nunique()
        total_percent = (unique_vals / ln) * 100
        eff_percent = (data[i].dropna().nunique() / ln) * 100
        avg_percent = (total_percent + eff_percent) / 2
        if avg_percent <= th:
            dtb.append(
                [
                    i,
                    round(unique_vals, 5),
                    round(total_percent, 5),
                    round(eff_percent, 5),
                    round(avg_percent, 5),
                ]
            )

    print(
        tb(
            dtb,
            headers=[
                "Column name",
                "Number of unique values",
                "Total uniqueness percent",
                "Effective uniqueness percent",
                "Average uniqueness percentage",
            ],
            tablefmt="fancy_grid",
        )
    )


def suggest_drops(data, th=60):
    dtb = []
    print(
        "Following columns might be considered to be dropped as percent of missing values are greater than the threshold-",
        th,
        "%\nLength of the dataset is:",
        len(data),
    )
    ln = len(data)

    for i in data.columns:
        nans = data[i].isna().sum()
        nan_percent = (nans / ln) * 100
        if nan_percent >= th:
            dtb.append([i, round(nans, 5), round(nan_percent, 5)])

    print(
        tb(
            dtb,
            headers=["Column name", "Number of nulls", "Percent of null values"],
            tablefmt="fancy_grid",
        )
    )


def suggest_fillers(data, th=40):
    dtb = []
    print(
        "Following columns might be considered to be imputed as percent of missing values are less than the threshold-",
        th,
        "%\nLength of the dataset is:",
        len(data),
    )
    ln = len(data)

    for i in data.columns:
        nans = data[i].isna().sum()
        nan_percent = (nans / ln) * 100
        if nan_percent <= th and nan_percent != 0:
            dtb.append([i, round(nans, 5), round(nan_percent, 5)])

    print(
        tb(
            dtb,
            headers=["Column name", "Number of nulls", "Percent of null values"],
            tablefmt="fancy_grid",
        )
    )
