import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import  preprocessing
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

fieldName = ['prediction', 'AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict', 'AvgEssential', 'AvgLine',
             'AvgLineBlank', 'AvgLineCode', 'AvgLineComment', 'AvgClassBase', 'AvgClassCoupled', 'AvgClassDerived',
             'CountDeclClass', 'CountDeclClassMethod', 'CountDeclClassVariable', 'Ca', 'CountDeclFunction',
             'CountDeclInstanceMethod', 'CountDeclInstanceVariable', 'CountDeclMethod', 'AvgDeclMethodAll',
             'CountDeclMethodDefault', 'CountDeclMethodPrivate', 'CountDeclMethodProtected', 'CountDeclMethodPublic',
             'CountLine', 'CountLineBlank', 'CountLineCode', 'CountLineCodeDecl', 'CountLineCodeExe',
             'CountLineComment', 'CountSemicolon', 'CountStmt', 'CountStmtDecl', 'CountStmtExe', 'MaxCyclomatic',
             'MaxCyclomaticModified', 'MaxCyclomaticStrict', 'MaxEssential', 'MaxInheritanceTree', 'MaxNesting',
             'AvgPercentLackOfCohesion', 'RatioCommentToCode', 'SumCyclomatic', 'SumCyclomaticModified',
             'SumCyclomaticStrict', 'SumEssential']
datasize = 27738
featuresize = len(fieldName)-1
hide1size = 5
rate = 0.001
epoch = 100000
# JavaSampling / JavaExp
dataDir=r"D:\科研\CodeQualityAnalysis\CodeAnalysis\JavaSampling\_analysisML\CodeQualityData.csv"

def getData(dataDir=dataDir, fieldName=fieldName):
    datadf = pd.read_csv(dataDir)[fieldName].sort_values(by='prediction')
    # filter out files with score <= 2
    datadf = datadf[datadf['prediction'] > 2]

    X = datadf.ix[:,1:]
    Y = datadf.ix[:,'prediction']

    print(Y.describe())
    xx = np.array(list(range(len(Y)))) + 1
    plt.figure()
    plt.plot(xx, Y.reshape(-1), 'bo')
    plt.show()


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42,shuffle=True)
    scalerX = preprocessing.StandardScaler().fit(X_train)
    X_train_std = scalerX.transform(X_train,copy=True)
    X_test_std = scalerX.transform(X_test,copy=True)
    return X_train_std, X_test_std, Y_train, Y_test

def linerRegression(train_x, train_y, epoch=epoch, rate=rate):
    train_x = np.array(train_x).reshape((-1,featuresize))
    train_y = np.array(train_y).reshape((-1,1))

    X = tf.placeholder('float64', [None, featuresize])
    Y = tf.placeholder('float64', [None, 1])
    weight1 = tf.Variable(tf.random_normal([featuresize, hide1size], stddev=0.35, dtype='float64'))
    biase1 = tf.Variable(tf.zeros([hide1size], dtype='float64'))
    Hide1 = tf.matmul(X, weight1) + biase1
    weight2 = tf.Variable(tf.random_normal([hide1size, 1], stddev=0.35, dtype='float64'))
    biase2 = tf.Variable(tf.zeros([1], dtype='float64'))
    y_pred = tf.matmul(Hide1,weight2) + biase2
    loss = tf.reduce_mean(tf.pow(y_pred-Y, 2))
    optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for index in range (epoch):
        sess.run(optimizer, {X:train_x, Y:train_y})
        if index % 20000 == 0:
            l = sess.run(loss, {X:train_x, Y:train_y})
            print("After %s steps,train_loss : %s" % (str(index),str(l)))
    w1, w2, b1, b2 = sess.run([weight1, weight2, biase1, biase2])
    predictionTest(train_x, train_y, w1, w2, b1, b2, "train")
    return w1, w2, b1, b2

def predictionTest(test_x,test_y,w1,w2,b1,b2, batch):
    print(batch + " performation: ")
    test_y = test_y.reshape(-1, 1)
    Hide = np.dot(test_x, w1) + b1
    pred = np.dot(Hide, w2) + b2
    loss = np.sqrt(np.mean(np.square(pred-test_y)))
    gragh = np.array((test_y.reshape(-1), pred.reshape(-1)))
    gragh = gragh.T[np.lexsort(gragh[::-1, :])].T
    xx = np.array(list(range(len(pred))))+1
    plt.figure()
    plt.title("pred vs true")
    plt.xlabel("file No.")
    plt.ylabel("score")
    plt.scatter(xx, gragh[0],c='blue',marker='o', label="true")
    plt.plot(xx, gragh[1], c='yellow',marker='x', label="pred")
    plt.show()
    print("R^2: ")
    print(r2_score(gragh[0], gragh[1]))
    print("MSE loss:")
    print(loss)
    return loss

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = getData()
    W1, W2, b1, b2 = linerRegression(X_train,Y_train)
    # loss = predictionTest(X_train,Y_train,W1,W2,b1,b2, "train")
    loss = predictionTest(X_test,Y_test,W1,W2,b1,b2, "test")
