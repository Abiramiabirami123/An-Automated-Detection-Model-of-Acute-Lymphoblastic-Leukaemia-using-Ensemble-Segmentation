import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle

no_of_dataset = 1

def Plot_ROC():
    lw=2
    cls = ['CNN', 'ResNet', 'VGG16', 'ResNet+DenseNet', 'OROA-LIH-DRNet']
    # Classifier = ['TERMS', 'Xgboost', 'DT', 'NN', 'FUZZY', 'KNN', 'PROPOSED']
    for a in range(no_of_dataset): # For 5 Datasets
        Actual = np.load('Target_1.npy', allow_pickle=True).astype('int')
        # Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')

        colors = cycle(["blue", "crimson", "gold", "lime", "black"]) #  "cornflowerblue","darkorange", "aqua"
        for i, color in zip(range(5), colors): # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_1_ROC_%s_.png" % (a+1)

        plt.savefig(path1)
        plt.show()
if __name__ =='__main__':
    Plot_ROC()