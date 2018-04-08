import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression

def getModel(X_train, y_train, modelName='linear', args=''):
    '''

    :return:
    '''
    if modelName == 'linear':
        linreg = LinearRegression(fit_intercept=True, normalize=True,copy_X=True, n_jobs=1)
        linreg.fit(X_train, y_train)
        return linreg
    else:
        print("model para wrong. modelName must be in ['linear']")
        return None

def predict(X_test, model):
    return model.predict(X_test)

def estimate(X_test, Y_test, model):
    y_pred = model.predict(X_test)
    MSE = metrics.mean_squared_error(Y_test, y_pred)
    RMSE = MSE**0.5
    print("model estimate: \nMSE: %f\nRMSE: %f" %(MSE, RMSE))
    print("model.score: ")
    print(model.score(X_test, Y_test))
    return MSE, RMSE


def showGragh(xaxi, Y_test, y_predicts, labels=['']):
    # y_predicts = [y_model1_pred, y_model2_pred, ...]
    COLOR = ['red', 'blue', 'yellow', 'green', 'gray', 'pink']
    plt.scatter(xaxi, Y_test, color='black', linewidths=0.2, label='real data')
    for i in range(len(y_predicts)):
        plt.scatter(xaxi, y_predicts[i],color=COLOR[i % len(COLOR)], linewidths=0.05, label=labels[i % len(labels)])
    plt.xlabel('file No.')
    plt.ylabel('stars')
    plt.show()