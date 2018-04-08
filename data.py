import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from  sklearn.linear_model import LinearRegression
from  sklearn.decomposition import PCA
import sys
import matplotlib.pyplot as plt

class CQdata(object):
    # dataDir: JavaSampling / JavaExp
    def __init__(self, dataDir=r"D:\科研\CodeQualityAnalysis\CodeAnalysis\JavaSampling\_analysisML\CodeQualityData.csv", fieldName=['prediction','AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict', 'AvgEssential', 'AvgLine', 'AvgLineBlank', 'AvgLineCode', 'AvgLineComment', 'AvgClassBase', 'AvgClassCoupled', 'AvgClassDerived', 'CountDeclClass', 'CountDeclClassMethod', 'CountDeclClassVariable', 'Ca', 'CountDeclFunction', 'CountDeclInstanceMethod', 'CountDeclInstanceVariable', 'CountDeclMethod', 'AvgDeclMethodAll', 'CountDeclMethodDefault', 'CountDeclMethodPrivate', 'CountDeclMethodProtected', 'CountDeclMethodPublic', 'CountLine', 'CountLineBlank', 'CountLineCode', 'CountLineCodeDecl', 'CountLineCodeExe', 'CountLineComment', 'CountSemicolon', 'CountStmt', 'CountStmtDecl', 'CountStmtExe', 'MaxCyclomatic', 'MaxCyclomaticModified', 'MaxCyclomaticStrict', 'MaxEssential', 'MaxInheritanceTree', 'MaxNesting', 'AvgPercentLackOfCohesion', 'RatioCommentToCode', 'SumCyclomatic', 'SumCyclomaticModified', 'SumCyclomaticStrict', 'SumEssential'], testsize=0.3, randstate=42, shuffle=True):
        self.dataDir = dataDir
        self.fieldName = fieldName
        self.datadf = pd.read_csv(dataDir)[fieldName].sort_values(by='prediction')
        # filter out files with score <= 2
        self.datadf = self.datadf[self.datadf['prediction'] > 2]

        # print("data describe:")
        # self.dataReguDf = preprocessing.scale(self.datadf)
        self.X = self.datadf.ix[:,1:]
        self.Y = self.datadf.ix[:,'prediction']

        print(self.Y.describe())
        xx = np.array(list(range(len(self.Y))))+1
        plt.figure()
        plt.plot(xx, self.Y.reshape(-1), 'bo')
        plt.show()

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=testsize, random_state=randstate,shuffle=shuffle)
        # return self.X_train, self.X_test, self.Y_train, self.Y_test
        self.scalerX = preprocessing.StandardScaler().fit(self.X_train)
        self.X_train_std = self.scalerX.transform(self.X_train,copy=True)
        self.X_test_std = self.scalerX.transform(self.X_test,copy=True)

# __name__ = '__showfeature__'
if __name__ == '__showfeature__':
    import numpy as np
    K = [1, 3, 5, 10, 15, 30]
    f = open(r'D:\科研\CodeQualityAnalysis\CodeAnalysis\aaa.txt', 'w')
    old = sys.stdout
    sys.stdout = f
    cqdata = CQdata(testsize=0.01, shuffle=False)

    # data = cqdata.originalData(show=False)
    # cordf = data.corr()
    # covdf = data.cov()
    # for ser in cordf:
    # for ser in covdf:
    #    print('')
        # for x in cordf[ser].values:
    #    for x in covdf[ser].values:
    #        print(str(x)+',',end='')

    y_pre = cqdata.Y_train
    xx = pd.Series(list(range(len(y_pre))))
    for i in range(1,len(cqdata.fieldName)):
        field = cqdata.fieldName[i]
        y_feature = cqdata.X_train_std[:,i-1]
        plt.subplot(2,2,(i-1) % 4 +1)
        plt.scatter(xx,y_pre,color='blue')
        plt.plot(xx, y_feature, 'rx')
        plt.title(field)
        i += 1
        if (i-1) % 4 == 0:
            plt.show()
    f.close()
    sys.stdout = old
    plt.show()

if __name__ == '__main__':
    cqdata = CQdata()
    X_train_std = cqdata.X_train_std
    X_test_std = cqdata.X_test_std
    Y_train = cqdata.Y_train
    Y_test = cqdata.Y_test
    for k in range(1, 47):
        pca = PCA(n_components=k)
        pca.fit(X_train_std)
        model = LinearRegression()
        X_train_pca = pca.transform(X_train_std)
        X_test_pca = pca.transform(X_test_std)
        model.fit(X_train_pca, Y_train)
        loss1 = np.sqrt(np.mean(np.square(model.predict(X_train_pca) - Y_train)))
        loss2 = np.sqrt(np.mean(np.square(model.predict(X_test_pca) - Y_test)))
        print("K=%d\tR^2 \tRMSE" %k)
        print("train\t" + str(model.score(X_train_pca, Y_train))+"\t"+str(loss1))
        print("test\t" + str(model.score(X_test_pca, Y_test))+"\t"+str(loss2))