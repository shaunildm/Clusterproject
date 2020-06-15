import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

## K-MEANS
print("\nPart 1 - Kmeans")
# 1. Apply k-means clustering to the articles.pkl
os.chdir('../data')
articles_df = pd.read_pickle("articles.pkl")
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(articles_df['content'])
features = vectorizer.get_feature_names()
kmeans = KMeans()
kmeans.fit(X)


# 2. Print out the centroids.
print("\n2) cluster centers:")
print(kmeans.cluster_centers_)


# 3. Find the top 10 features for each cluster.
top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
print("\n3) top features (words) for each cluster:")
for num, centroid in enumerate(top_centroids):
    print("%d: %s" % (num, ", ".join(features[i] for i in centroid)))

    
# 4. Limit the number of features and see if the words of the topics change.
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(articles_df['content'])
features = vectorizer.get_feature_names()
kmeans = KMeans()
kmeans.fit(X)
top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
print("\n4) top features for each cluster with 1000 max features:")
for num, centroid in enumerate(top_centroids):
    print("%d: %s" % (num, ", ".join(features[i] for i in centroid)))

    
# 5. Print out the titles of a random sample of the articles assigned to each
# cluster to get a sense of the topic.
print("\n5) random sample of titles in each cluster")
assigned_cluster = kmeans.transform(X).argmin(axis=1)
for i in range(kmeans.n_clusters):
    cluster = np.arange(0, X.shape[0])[assigned_cluster==i]
    sample_articles = np.random.choice(cluster, 3, replace=False)
    print("cluster %d:" % i)
    for article in sample_articles:
        print("    %s" % articles_df.ix[article]['headline'])


# 6. Which topics has k-means discovered?

# 7. If you set k == to the number of NYT sections in the dataset, does it
# return topics that map to a section?
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
assigned_cluster = kmeans.transform(X).argmin(axis=1)
print("\n7) top 2 topics for each cluster")
for i in range(kmeans.n_clusters):
    cluster = np.arange(0, X.shape[0])[assigned_cluster==i]
    topics = articles_df.ix[cluster].dropna()['section_name']
    most_common = Counter(topics).most_common()
    print("Cluster %d: %s" % (i, most_common[0][0]))
    if len(most_common) > 1:
        print(" %s" % (most_common[1][0]))

# It sort of classifies by section, but some (like sports and world) end up in
# multiple clusters.


# 8. Try clustering with a subset of the sections.
mask = np.logical_or(
       np.logical_or((articles_df['section_name']=='Sports').values,
                     (articles_df['section_name']=='Arts').values),
                     (articles_df['section_name']=='Business Day').values)
three_articles_df = articles_df[mask]
kmeans = KMeans(n_clusters=3)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(three_articles_df['content'])
kmeans.fit(X)
assigned_cluster = kmeans.transform(X).argmin(axis=1)
print("8) Top 2 topics for each cluster")
for i in range(kmeans.n_clusters):
    cluster = np.arange(0, X.shape[0])[assigned_cluster==i]
    topics = three_articles_df.ix[cluster].dropna()['section_name']
    most_common = Counter(topics).most_common()
    print("Cluster %d: %s" % (i, most_common[0][0]))
    if len(most_common) > 1:
        print(" %s" % (most_common[1][0]))


