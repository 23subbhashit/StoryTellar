from collections import Counter
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tabulate import tabulate as tb
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import re
from scipy.cluster.hierarchy import linkage, dendrogram
from nltk.stem.snowball import SnowballStemmer
from datetime import date
from sklearn.cluster import KMeans
from wordcloud import WordCloud

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
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer


def subbhashit():
    return "Hi Vro"


def shree():
    return "HI SHREE"


def shivang():
    return "HI GUJJU"


def count(x):
    array = list(x)
    countArray = dict(Counter(array))
    return countArray


def impute(array, method="mean"):
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


def zscore(data, threshold=1):
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


def singleplot(arr):
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12.7, 10.27)

    plt.subplot(2, 2, 1)
    arr.value_counts().tail().plot(kind="pie", figsize=(15, 10))

    sns.distplot(arr, ax=ax[0, 1])

    plt.subplot(2, 2, 3)
    arr.value_counts().tail().plot(kind="bar", color=["c", "y", "r"], figsize=(15, 10))

    sns.boxplot(arr, ax=ax[1, 1])

    fig.show()


def iqr(data, arg1=75, arg2=25):
    q3, q1 = np.percentile(data, [arg1, arg2])
    iqr = q3 - q1
    return iqr


def describe(data):
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


def chloropleth(data, title="", hue=""):
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


def error_score(yt, yp, typ="classification", beta=0.5, average="macro"):
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
    else:
        return "Enter a valid type"


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

    return pd.DataFrame(
        dtb,
        columns=[
            "Column name",
            "Number of unique values",
            "Total uniqueness percent",
            "Effective uniqueness percent",
            "Average uniqueness percentage",
        ],
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

    return pd.DataFrame(
        dtb, columns=["Column name", "Number of nulls", "Percent of null values"]
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

    return pd.DataFrame(
        dtb, columns=["Column name", "Number of nulls", "Percent of null values"]
    )


# function for parsing datetime
def formatted_date(df):
    for col in df.columns:
        if col == "date" or col == "Date":
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
            print(df[col])


# Function for cleaning of texts
def process_text(x):
    processed_tweet = re.sub(r"\W", " ", str(x))
    processed_tweet = re.sub(r"\s+[a-zA-Z]\s+", " ", processed_tweet)
    processed_tweet = re.sub(r"\^[a-zA-Z]\s+", " ", processed_tweet)
    processed_tweet = re.sub(r"\s+", " ", processed_tweet, flags=re.I)
    processed_tweet = re.sub(r"^b\s+", "", processed_tweet)
    processed_tweet = processed_tweet.lower()
    return processed_tweet


def tfidf_vectorizer(x, max_featues=1000, min_df=5, max_df=0.7):
    tfidfconverter = TfidfVectorizer(
        max_features=max_featues,
        min_df=min_df,
        max_df=max_df,
        stop_words=stopwords.words("english"),
    )
    df = tfidfconverter.fit_transform(x).toarray()
    return df


def get_cosine_dict(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_dict(text):
    WORD = re.compile(r"\w+")
    words = WORD.findall(text)
    return Counter(words)


def similarity_matrix(sentences):
    "gives a matrix for sentence similarity"
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for index1 in tqdm(range(len(sentences))):
        for index2 in range(len(sentences)):
            if index1 == index2:
                continue
            similarity_matrix[index1][index2] = get_cosine_dict(
                sentences[index1], sentences[index2]
            )
    return similarity_matrix


def cosine_distance_vector(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    v1 = list(v1)
    v2 = list(v2)
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def suggest_similar(df, unique_id, col):

    stemmer = SnowballStemmer("english")

    def tokenize_and_stem(text):

        tokens = [
            word
            for sentence in nltk.sent_tokenize(text)
            for word in nltk.word_tokenize(sentence)
        ]

        filtered_tokens = [token for token in tokens if re.search("[a-zA-Z]", token)]

        stems = [stemmer.stem(token) for token in filtered_tokens]

        return stems

    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.8,
        max_features=200000,
        min_df=0.2,
        stop_words="english",
        use_idf=True,
        tokenizer=tokenize_and_stem,
        ngram_range=(1, 3),
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in df[col]])

    km = KMeans(n_clusters=5)

    km.fit(tfidf_matrix)

    clusters = km.labels_.tolist()

    df["cluster"] = clusters

    similarity_distance = 1 - cosine_similarity(tfidf_matrix)

    mergings = linkage(similarity_distance, method="complete")

    dendrogram_ = dendrogram(
        mergings,
        labels=[x for x in df[unique_id]],
        leaf_rotation=90,
        leaf_font_size=16,
    )

    fig = plt.gcf()

    _ = [lbl.set_color("r") for lbl in plt.gca().get_xmajorticklabels()]

    fig.set_size_inches(108, 21)

    plt.show()


def catvscatplot(arr1, arr2, stacked=True):
    b = pd.crosstab(arr1, arr2)
    b.tail(10).plot.bar(stacked=stacked, figsize=(15, 9))


def catvsnumericalplot(
    data,
    catcol,
    numcol,
    stacked=True,
    swarmcolor="c",
    violincolor="r",
    kdecolor="y",
    scattercolor="b",
    linecolor="g",
):
    # Plots initialization
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12.7, 10.27)

    # Scatterplot
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=catcol, y=numcol, data=data, color=scattercolor)

    # Swarm+Violin plot
    plt.subplot(2, 2, 2)
    sns.swarmplot(x=catcol, y=numcol, data=data, color=swarmcolor)
    sns.violinplot(x=catcol, y=numcol, data=data, color=violincolor)

    # Bar plot
    plt.subplot(2, 2, 3)

    sns.barplot(x=catcol, y=numcol, data=data)

    # Box plot
    plt.subplot(2, 2, 4)
    sns.boxplot(x=catcol, y=numcol, data=data)

    #     t=data.pivot_table(index=catcol,values=numcol,aggfunc=np.median)
    #     t.plot(kind="bar",color=['c','y','r'])

    fig.show()


def numvsnumplot(arr1, arr2, stacked=True, scattercolor="c", linecolor="r"):
    # Plots initialization
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12.7, 5.27)

    # Scatterplot
    plt.subplot(1, 2, 1)
    sns.scatterplot(arr1, arr2, color=scattercolor)

    # Lineplot
    plt.subplot(1, 2, 2)
    sns.lineplot(arr1, arr2, color=linecolor)

    fig.show()


def suggest_quants(data, th=60):
    dtb = []
    print(
        "Following columns might be considered to be converted as categorical as \nthe column is numerical and the uniqueness percent is greater than the threshold-",
        th,
        "%\nLength of the dataset is:",
        len(data),
    )
    ln = len(data)
    numer = data.select_dtypes(include=np.number).columns.tolist()

    for i in numer:
        unique_vals = data[i].nunique()
        total_percent = (unique_vals / ln) * 100
        if total_percent >= 60:
            dtb.append([i])

    print(tb(dtb, headers=["Column name"], tablefmt="fancy_grid"))


def create_quants(data, cols):
    dtb = []
    print("Creating Quantile columns...")

    for col in cols:
        low = np.percentile(data[col], 25)
        mid = np.percentile(data[col], 50)
        high = np.percentile(data[col], 75)
        data[col + "_quant"] = data[col].apply(
            lambda i: 0 if low > i else (1 if mid > i else (2 if high > i else 3))
        )
        print(col + "_quant" + " has been created using column " + col)

    print("completed!")


def date_me(data, cols):
    from datetime import date

    today = date.today()
    dtb = []
    print("Starting feature extraction from date column...")

    for col in cols:
        data["age"] = today.year - data[col].dt.year
        data["months"] = data["age"] * 12 + data[col].dt.month
        data["days"] = data["months"] * 30 + data[col].dt.day
        data["season"] = data["months"].apply(
            lambda i: "Winter"
            if i in [1, 2, 12]
            else (
                "Spring"
                if i in [4, 5, 6]
                else ("Summer" if i in [7, 8, 9] else "Spring")
            )
        )
        data["weekday"] = data[col].dt.day_name()
        print("Features Extracted from column", col + ".....")

    print("completed!")


def time_series_plot(data, datefrom, dateto, text, col, figsize=(16, 9)):
    data[col][datefrom:].plot(figsize=figsize, legend=True, color="r")
    data[col][:dateto].plot(figsize=figsize, legend=True, color="g")
    title1 = "Data (Before {})".format(datefrom)
    title2 = "Data {} and beyond)".format(dateto)
    plt.legend([title1, title2])
    plt.title(text)
    plt.show()


def wordarraycloud(arr_words, width=800, height=400):
    states = np.array(arr_words)
    cloud = WordCloud(width=width, height=height)
    cloud.generate(" ".join(states))
    return cloud.to_image()