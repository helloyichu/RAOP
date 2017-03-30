from __future__ import division
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.cluster import KMeans
import enchant

from util import raopUtil




# create data frames like dictionary {column -> {index -> value}}
# series is 1 dimension, data frame is 2 dimension, panel is 3 dimension
training = pd.read_json('train.json', orient="columns")
testing = pd.read_json('test.json', orient='columns')
trueScaling = 3
falseScaling = 1

vect = CountVectorizer()




# create arrays with structure [index][column]
np_test = np.array(testing)
np_train = np.array(training)
mainTextInTest = np_test[:,2]
textIDInTest = np_test[:,1]
X = np_train[:]
Y = np_train[:,22]

# get (rows amount , columns amount) tuple
# tuple is immutable, but can index like list


# get all main texts

# whether requester receive pizza or not
# numpy.ndarrays can be indexed by [rows, columns]
# whereby rows and columns are array as well,
# a match between rows and columns will be used to produce the ndarray
# the size of ndarray produced = sizeof(rows) , sizeof(columns)


actualFalse = []
actualTrue = []

for i in range(len(X)):
    if Y[i]:
        actualTrue.append(i)
    else:
        actualFalse.append(i)



X_true, Y_true = X[[value for value in actualTrue for _ in range(trueScaling)]] , [True for value in actualTrue for _ in range(trueScaling)]
X_false, Y_false = X[[value for value in actualFalse for _ in range(falseScaling)]] , [False for value in actualFalse for _ in range(falseScaling)]

X= np.concatenate((X_true, X_false))
Y= np.concatenate((Y_true,Y_false))

# randomly permutation take an array object and shuffle it by its first index
shuffle = np.random.permutation(np.arange(X.shape[0]))

X , Y = X[shuffle] , Y[shuffle]


l = len(X)

print training.columns

training, train_labels = X[:3 * l / 4], Y[:3 * l / 4]
dev, dev_labels= X[3 * l / 4:] , Y[3 * l / 4:]




textTraining = training[: , 6]
textDev = dev[:,6]
#print training
print (textTraining.shape)
#data = vect.fit_transform(train_data).toarray()
# learn the vocabulary dictionary and return a document term matrix with each entry in value of counts

#devdata = vect.transform(dev_data).toarray()
# transform using existing vocabulary

#print vect.get_feature_names()


print "Set up datasets and benchmark"
print "Training set shape: " ,np_train.shape
print "Testing set shape: ", np_test.shape
print "Scaled training shape: ", training.shape
print "Dev training shape: ", dev.shape

b=train_labels
train_labels = np.where(b == True , 1,0)

b2 = dev_labels
dev_labels = np.where(b2 == True, 1, 0)


count=0
for k in range(len(Y)):
    if Y[k]== True:
        count+=1
print 'proportion of success in the original dataset', count / len(Y)



count=0
for k in range(len(dev_labels)):
    if dev_labels[k]!=0:
        count+=1
print 'if predict all as 0 , accuracy will be', 1 - count / len(dev_labels)


###################################################


print ""
print "Build text analyzer"
kmeans=raopUtil.clusterLearning(textTraining)
cluster = np.array(kmeans.labels_)

clusterDev=kmeans.predict(textDev)
###################################################
#add remaining variables
#number of comments: 5
#requester_upvotes_plus_downvotes_at_request',: 26
#requester_upvotes_plus_downvotes_at_retrieval',: 27
#requester_account_age_in_days_at_request : 9
dataForTraining=[]
for i in range(len(training)):
    dataForTraining.append([int(training[i ,5]),int(training[i ,9]),int(training[i ,26])
                    ,int(training[i ,27]),cluster[i]])

dataForTesting =[]

#for i in range(len(training)):
#   dataForTesting.append([int(training[i ,5]),int(training[i ,9]),int(training[i ,26])
#                   ,int(training[i ,27]),cluster[i]])



#multinomialNB classifier
# alpha is the parameter for addictive smoothing ,
# avoid P(Xi = xij | Y= yi) =0 and the naive probability will be calculated as 0
#######################
best_nb = []


alphas = [ 0.0001, 0001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
for k in range(len(alphas)):
    mnb_classifier = Pipeline([('mnclf', MultinomialNB(alpha= alphas[k]))])
    mnb_classifier = mnb_classifier.fit(dataForTraining, train_labels)
    prediction = mnb_classifier.predict(textDev)
    score = metrics.accuracy_score(dev_labels, prediction)
    best_nb.append(score)
bestAlphaAccuracy = max(best_nb)
bestAlphaValue = alphas[best_nb.index(bestAlphaAccuracy)]
print 'Naive bayes finding best alpha'
print'Best Alpha = ', bestAlphaValue
print 'accuracy = ', bestAlphaAccuracy
print ''

print 'Naive Bayes prediction'

mnb_classifier = Pipeline([('vect', CountVectorizer()),('mnclf', MultinomialNB(alpha= bestAlphaValue))])
mnb_classifier = mnb_classifier.fit(textTraining, train_labels)
prediction = mnb_classifier.predict(mainTextInTest)


count = 0
result =[]
for i in range(len(prediction)):
    result.append([textIDInTest[i], prediction[i]])
    if prediction[i] !=0:
        count+=1

print "predicted result not equal to 0 : " , count

print "prediction result "
print result

result = np.asarray(result)

df = pd.DataFrame(result)
df.to_csv("result.csv", index=False)

#np.savetxt("result.csv" , result, delimiter=",")














