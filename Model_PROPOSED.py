from Evaluation import evaluation
from Model_DenseNet import Model_DenseNet
from Model_Resnet import Model_RESNET


def model_proposed(Train_Data, Train_Target, Test_data, Test_Target, sol=None):
    if sol is None:
        sol = [5, 5, 3, 5, 5, 3]
    Eval1, pred1 = Model_DenseNet(Train_Data, Train_Target, Test_data, Test_Target,sol[:3])
    Eval2, pred2 = Model_RESNET(Train_Data, Train_Target, Test_data, Test_Target,sol[3:])

    pred = ((pred1 + pred2) / 2).astype('int')
    Eval = evaluation(pred, Test_Target)
    return pred, Eval


