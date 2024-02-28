import numpy as np
import cv2 as cv
from Model_PROPOSED import model_proposed
from Region_Growing import Region_Growing
from FCM import FCM
from Global_Vars import Global_Vars
from mutual_information import MutualInfoImage


def Obj_seg(Soln):
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for k in range(v):
        soln = np.array(Soln).astype(np.uint8)
        if soln.ndim == 2:
            sol = np.round(soln[k, :])
        else:
            sol = np.round(soln)

        sol = sol.astype(int)
        Preprocess = Global_Vars.Feat
        gro_tru = Global_Vars.Tar
        analysis = cv.connectedComponentsWithStats(Preprocess, 4, cv.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis
        for i in range(len(sol)):
            if sol[i] == 0:
                Preprocess[label_ids == i + 1] = 0
        mutual = MutualInfoImage(Preprocess, gro_tru)
        Fitn[k] = 1 / mutual  # Maximization of Accuracy
    return Fitn
