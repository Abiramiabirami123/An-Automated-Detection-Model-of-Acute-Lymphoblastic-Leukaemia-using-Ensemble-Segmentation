import warnings
import seaborn as sn
import pandas as pd
from sklearn.metrics import roc_curve
from itertools import cycle
import matplotlib
from prettytable import PrettyTable
from Image_Results import Image_Results_seg
from Plot_ROC import Plot_ROC
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt


no_of_dataset = 1

### For learning percentage
def plot_results():
    eval1 = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']

    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Algorithm = ['TERMS', 'PSO-LIH-DRNet', 'JAYA-LIH-DRNet', 'GSO-LIH-DRNet', 'ROA-LIH-DRNet', 'OROA-LIH-DRNet']
    Classifier = ['TERMS', 'CNN', 'ResNet', 'VGG16', 'Resnet+Densenet', 'OROA-LIH-DRNet']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 4, :, 4:]



    learnper = [45, 55, 65, 75, 85]
    for i in range(eval1.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros((eval1.shape[1], eval1.shape[2] + 1))
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                        if j == 10:
                            Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]
                        else:
                            Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]

            plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                     label="PSO-LIH-DRNet")
            plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                     label="JAYA-LIH-DRNet")
            plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                     label="GSO-LIH-DRNet")
            plt.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                     label="ROA-LIH-DRNet")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='white',markersize=12,
                     label="OROA-LIH-DRNet")
            # plt.plot(learnper, Graph[:, 5], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
            #          label="GRSO-CRN")
            plt.xticks(learnper, ('0.01', '0.11', '0.21', '0.31', '0.41'))
            plt.xlabel('Learning rate')
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc=4)
            path1 = "./Results/Dataset_%s_%s_line_lrean.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)

            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="CNN")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="ResNet")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="VGG16")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="Resnet+Densenet")
            # ax.bar(X + 0.40, Graph[:, 9], color='c', width=0.10, label="ENSEMBLE")
            ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="OROA-LIH-DRNet")
            plt.xticks(X + 0.10, ('0.01', '0.11', '0.21', '0.31', '0.41'))
            plt.xlabel('Learning rate')
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc=4)
            path1 = "./Results/Dataset_%s_%s_bar_lrean.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

def plot_results_table():
    eval1 = np.load('Eval_all_tb.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Algorithm = ['TERMS', 'PSO-LIH-DRNet', 'JAYA-LIH-DRNet', 'GSO-LIH-DRNet', 'ROA-LIH-DRNet', 'OROA-LIH-DRNet']
    Classifier = ['TERMS', 'CNN', 'ResNet', 'VGG16', 'ResNet+DenseNet', 'OROA-LIH-DRNet']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- ' 'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- ' 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v

def plot_results_conv():
    # matplotlib.use('TkAgg')
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'PSO-LIH-DRNet', 'JAYA-LIH-DRNet', 'GSO-LIH-DRNet', 'ROA-LIH-DRNet', 'OROA-LIH-DRNet']

    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
            # a = 1
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- ''Statistical Report ',
              '--------------------------------------------------')

        print(Table)

        length = np.arange(25)
        Conv_Graph = Fitness[i]

        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red', markersize=12,
                 label='PSO-LIH-DRNet')
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12,
                 label='JAYA-LIH-DRNet')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='cyan',
                 markersize=12,
                 label='GSO-LIH-DRNet')
        plt.plot(length, Conv_Graph[3, :], color='y', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12,
                 label='ROA-LIH-DRNet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12,
                 label='OROA-LIH-DRNet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/User Variation_%s_%s_Conv.png" % (i + 1, 1))
        plt.show()



def plot_confusion():
    for i in range(1): # For 1 datasets
        Eval = np.load('Eval_all_tb.npy', allow_pickle=True)[i]
        value = Eval[4, 4, :5]
        val = np.asarray([0, 1, 1])
        data = {'y_Actual': [val.ravel()],
                'y_Predicted': [np.asarray(val).ravel()]
                }
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'], colnames=['Predicted'])
        value = value.astype('int')


        confusion_matrix.values[0, 0] = value[1]  # -10700
        confusion_matrix.values[0, 1] = value[3]
        confusion_matrix.values[1, 0] = value[2]  # -3852
        confusion_matrix.values[1, 1] = value[0]

        sn.heatmap(confusion_matrix, fmt='d', annot=True).set(title='Accuracy = '+str(Eval[4, 4, 4]*100)[:5]+'%')
        sn.plotting_context()
        path1 = './Results/Confusion_'+str(i+1)+'.png'
        plt.savefig(path1)
        plt.show()



if __name__ == '__main__':
    plot_results()
    plot_results_table()
    plot_confusion()
    plot_results_conv()
    Plot_ROC()
    Image_Results_seg()



