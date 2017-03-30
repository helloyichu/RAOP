from __future__ import division
import enchant
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.cluster import KMeans

def clearTextFeatures(textFeatures):
    filter = []
    d = enchant.Dict("en_US")

    for i in range(len(textFeatures)):
        if(str.isalpha(textFeatures[i].encode('ascii','ignore'))):
            textFeatures[i]=textFeatures[i].encode('ascii','ignore')
            if d.check(textFeatures[i]):
                filter.append(i)

    return (textFeatures[filter], filter)

def clusterLearning(textArray):

    cv = CountVectorizer()

    countVectors = np.array(cv.fit_transform(textArray).toarray())
    textFeatures = np.array(cv.get_feature_names())

    print "Original count vectors shape ", countVectors.shape

    a=clearTextFeatures(textFeatures)
    textFeatures = a[0]
    filter = a[1]

    print textFeatures
    print textFeatures.shape

    updatedCountVectors = []
    for i in range(len(countVectors)):
        updatedCountVectors.append(countVectors[i][filter])



    updatedCountVectors = np.array(updatedCountVectors)

    print updatedCountVectors.shape

    kmeans = KMeans(n_clusters=4,random_state=0).fit(updatedCountVectors)


    clusterMeans = kmeans.cluster_centers_
    for i in range(len(clusterMeans)):
        f1= max(clusterMeans[i])
        f2= min(clusterMeans[i])
        for j in range(len(clusterMeans[i])):
            if(clusterMeans[i][j]==f1):
                print "Cluster ", i , " ", textFeatures[j]
            if (clusterMeans[i][j] == f2):
                print "Cluster ", i, " ", textFeatures[j]

    return kmeans
