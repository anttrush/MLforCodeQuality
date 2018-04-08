# import matplotlib.cm
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DrawPcolor(object):
    def __init__(self):
        ##self define the colorbar
        startcolor = '#006400'  # a dark green
        midcolor = '#ffffff'  # a bright white
        endcolor = '#ee0000'  # a dark red
        self.Mycmap = col.LinearSegmentedColormap.from_list('MyColorbar', [startcolor, midcolor,
                                                                           endcolor])  # use the "fromList() method

    def Pcolor(self, *args, **kwargs):
        # *args is a tuple,**kwargs is a dict;
        # Here args means the Matrix Corr,kwargs includes the key of the " AddText function"
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.Data = args[0]
        heatmap = self.ax.pcolor(self.Data, cmap=self.Mycmap)  # cmap=plt.cm.Reds)
        self.fig.colorbar(heatmap)
        # want a more natural, table-like display
        self.ax.invert_yaxis()
        self.ax.xaxis.tick_top()
        self.ax.set_xticks(np.arange(self.Data.shape[0]) + 0.5, minor=False)
        self.ax.set_yticks(np.arange(self.Data.shape[1]) + 0.5, minor=False)

        if kwargs['AddText'] == True:
            for y in range(self.Data.shape[1]):
                for x in range(self.Data.shape[0]):
                    self.ax.text(x + 0.5, y + 0.5, '%.2f' % self.Data[y, x],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 )
        self.fig.show()

    def Set_labelticks(self, *tick_labels):
        # put the major ticks at the middle of each cell
        self.ax.set_xticklabels(tick_labels[0], rotation=0, minor=False)
        self.ax.set_yticklabels(tick_labels[1], rotation=0, minor=False)
        self.fig.show()


def Main():
    fieldName = ['prediction', 'AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict', 'AvgEssential',
                 'AvgLine',
                 'AvgLineBlank', 'AvgLineCode', 'AvgLineComment', 'AvgClassBase', 'AvgClassCoupled', 'AvgClassDerived',
                 'CountDeclClass', 'CountDeclClassMethod', 'CountDeclClassVariable', 'Ca', 'CountDeclFunction',
                 'CountDeclInstanceMethod', 'CountDeclInstanceVariable', 'CountDeclMethod', 'AvgDeclMethodAll',
                 'CountDeclMethodDefault', 'CountDeclMethodPrivate', 'CountDeclMethodProtected',
                 'CountDeclMethodPublic',
                 'CountLine', 'CountLineBlank', 'CountLineCode', 'CountLineCodeDecl', 'CountLineCodeExe',
                 'CountLineComment', 'CountSemicolon', 'CountStmt', 'CountStmtDecl', 'CountStmtExe', 'MaxCyclomatic',
                 'MaxCyclomaticModified', 'MaxCyclomaticStrict', 'MaxEssential', 'MaxInheritanceTree', 'MaxNesting',
                 'AvgPercentLackOfCohesion', 'RatioCommentToCode', 'SumCyclomatic', 'SumCyclomaticModified',
                 'SumCyclomaticStrict', 'SumEssential']
    dataDir = r"D:\科研\CodeQualityAnalysis\CodeAnalysis\JavaSampling\_analysisML\CodeQualityData.csv"
    datadf = pd.read_csv(dataDir)[fieldName].sort_values(by='prediction')

    cordf = datadf.corr()
    corr = cordf.values

    xlabel_ticks = fieldName  # range(Corr.shape[0]);#
    ylabel_ticks = fieldName  # range(Corr.shape[1]);#
    MyPict = DrawPcolor()
    MyPict.Pcolor(corr, AddText=False)  # params 3,4,4,b has no effect on the exec
    MyPict.Set_labelticks(xlabel_ticks, ylabel_ticks)


if __name__ == '__main__':
    Main()