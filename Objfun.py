import numpy as np
import cv2 as cv
from scipy.ndimage import variance
from scipy.stats import entropy
from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_PROPOSED import model_proposed


def objfun_cls(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Tar
    Tar = np.reshape(Tar, (-1, 1))
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred_ = model_proposed(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            predict = pred_
            Eval = evaluation(predict, Test_Target)
            Fitn[i] = 1/(Eval[7] + Eval[13])
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred_ = model_proposed(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        predict = pred_
        Eval = evaluation(predict, Test_Target)
        Fitn = 1 / (Eval[7] + Eval[13])
        return Fitn




