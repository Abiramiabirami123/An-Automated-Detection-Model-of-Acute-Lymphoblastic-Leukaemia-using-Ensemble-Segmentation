import cv2 as cv
import os
import numpy as np
import random as rn
from FCM import FCM
from GSO import GSO
from Global_Vars import Global_Vars
from Image_Results import Image_Results_seg
from JAYA import JAYA
from Model_CNN import Model_CNN
from Model_PROPOSED import model_proposed
from Model_Resnet import Model_RESNET
from Model_VGG import Model_VGG16
from Obj_Cls import Obj_seg
from Objfun import objfun_cls
from PROPOSED import PROPOSED
from PSO import PSO
from Plot_ROC import Plot_ROC
from Plot_Results import plot_results, plot_results_table, plot_results_conv, plot_confusion
from ROA import ROA
from Region_Growing import Region_Growing

no_of_dataset = 1


def Read_data(dataset):
    Datas = []
    Target = []
    files = os.listdir(dataset)
    for i in range(len(files)):
        fold = dataset + files[i] + '/'
        sub_file = os.listdir(fold)
        data = []
        for j in range(len(sub_file)):
            print(i, j)
            images = fold + sub_file[j]
            img = cv.imread(images)  # to read the image file
            image = np.uint8(img)  # to change the unsigned int bit 8 value
            if len(image.shape) == 3:  # check whether image is RGB or not
                image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # changing the color model ie,RGB2GRAY
            image = cv.resize(image, (512, 512))  # for resizing image / scaling

            if 'Benign' == files[i]:
                target = 0
            elif 'Early' == files[i]:
                target = 1
            elif 'Pre' == files[i]:
                target = 2
            else:
                target = 3

            Target.append(target)
            Tar = np.reshape(Target, (-1, 1))
            Datas.append(image)
    return Datas, Tar


## Read the dataset
an = 0
if an == 1:
    Data, Target = Read_data('./Datasets/Dataset1/Original/')
    ground, target = Read_data('./Datasets/Dataset1/Segmented/')
    np.save('Images_1.npy', Data)
    np.save('Ground_Truth_1.npy', ground)
    np.save('Tar_1.npy', Target)

# arrange Target
an = 0
if an == 1:
    for n in range(no_of_dataset):
        tar = np.load('Tar_1.npy', allow_pickle=True)  # Load the dataset
        Target = np.asarray(tar)
        uniq = np.unique(Target)
        target = np.zeros((Target.shape[0], len(uniq))).astype('int')
        for uni in range(len(uniq)):
            print(n, uni)
            index = np.where(Target == uniq[uni])
            target[index[0], uni] = 1
        np.save('Target_1.npy', target)


##Preprocess contrast enhancement and Median filtering process
an = 0
if an == 1:
    for n in range(no_of_dataset):
        image = np.load('Images_1.npy', allow_pickle=True)
        pre = []
        for j in range(len(image)):
            print(n, j)
            images = image[j]
            alpha = 1.2  # Contrast control
            beta = 0.2  # Brightness control
            imagess = cv.convertScaleAbs(images, alpha=alpha, beta=beta)  # Contrast Enhancement
            img = cv.medianBlur(imagess, 5)  # Median Filtering
            pre.append(img)
        np.save('Preprocess_1.npy', pre)


# Segmentation
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Preprocess = np.load('Preprocess_1.npy', allow_pickle=True)
        segm = []
        for i in range(20):  # len(Preprocess)
            print('SEGMENTATION', i)
            image = Preprocess[i]
            cluster = FCM(image, image_bit=8, n_clusters=8, m=10, epsilon=0.8, max_iter=30)
            cluster.form_clusters()
            result = cluster.result.astype('uint8') * 30
            values, counts = np.unique(result, return_counts=True)
            index = np.where(counts == np.max(counts))[0][0]
            for j in range(len(values)):
                if j == index:
                    result[result == values[index]] = 0
                else:
                    result[result == values[j]] = 255

            analysis = cv.connectedComponentsWithStats(result, 4, cv.CV_32S)
            (totalLabels, label_ids, values, centroid) = analysis

            output = np.zeros(result.shape, dtype="uint8")
            # Loop through each component
            seed = []
            for i in range(1, totalLabels):
                ind = np.where(label_ids == label_ids[i])
                seed.append([ind[0][len(ind[0]) // 2], ind[1][len(ind[1]) // 2]])

            reg_grow = Region_Growing(result, seed)
            reg_gr = np.asarray(reg_grow)
            image_i = 1 - reg_gr
            regio = cv.resize(image_i, (512, 512))

            result[result != 0] = 1
            regio[regio != 0] = 1

            result = np.uint8(result)
            regio = np.uint8(regio)

            dest_and = cv.bitwise_and(result, regio, mask=None)

            output = np.zeros(image.shape, dtype=np.uint8)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            pixel_vals = image.reshape((-1, 3))
            pixel_vals = np.float32(pixel_vals)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            k = 3
            retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            img = segmented_data.reshape(image.shape).astype("uint8")
            res = img.astype('uint8') * 30
            uniq, counts = np.unique(res, return_counts=True)
            index1 = np.where(counts == np.max(counts))[0][0]
            for j in range(len(uniq)):
                if j == index1:
                    res[res == uniq[index1]] = 0
                else:
                    res[res == uniq[j]] = 255

            if len(res.shape) == 3:  # check whether image is RGB or not
                res = cv.cvtColor(res, cv.COLOR_RGB2GRAY)
            res[res != 0] = 1
            res = np.uint8(res)
            Ensem = cv.bitwise_and(dest_and, res, mask=None)
            Ensemble_Segmented = (Ensem * 255).astype('uint8')
            segm.append(Ensemble_Segmented)
        np.save('Segmentation_1.npy', segm)


# OPTIMIZATION for Optimal pixel selection
an = 0
if an == 1:
    Best_sol = []
    for n in range(no_of_dataset):
        Images = np.load('Segmentation_1.npy', allow_pickle=True)[:10]
        GT = np.load('Ground_Truth_1.npy', allow_pickle=True)[:10]
        bes_sol = []
        for i in range(len(Images)):
            analysis = cv.connectedComponentsWithStats(Images[i], 4, cv.CV_32S)
            (totalLabels, label_ids, values, centroid) = analysis

            Global_Vars.Feat = Images[i]
            Global_Vars.Tar = GT[i]
            Npop = 10
            Chlen = totalLabels - 1  # hidden neuron count ,Epochs a in Densenet
            xmin = np.zeros((Npop, Chlen))
            xmax = np.ones((Npop, Chlen))
            initsol = np.zeros(xmin.shape)
            for i in range(xmin.shape[0]):
                for j in range(xmin.shape[1]):
                    initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
            fname = Obj_seg
            max_iter = 50

            print('PSO....')
            [bestfit1, fitness1, bestsol1, Time1] = PSO(initsol, fname, xmin, xmax, max_iter)

            print('JAYA....')
            [bestfit2, fitness2, bestsol2, Time2] = JAYA(initsol, fname, xmin, xmax, max_iter)

            print('GSO....')
            [bestfit3, fitness3, bestsol3, Time3] = GSO(initsol, fname, xmin, xmax, max_iter)

            print('GSO....')
            [bestfit4, fitness4, bestsol4, Time4] = ROA(initsol, fname, xmin, xmax, max_iter)

            print('PROPOSED....')
            [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

            sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
            bes_sol.append(sol)
        Best_sol.append(bes_sol)
    np.save('Best_sol.npy', Best_sol)




# OPTIMIZATION FOR CLASSIFICATION
an = 0
if an == 1:
    Best_sol = []
    for n in range(no_of_dataset):
        Images = np.load('Segmentation_1.npy', allow_pickle=True)
        Target = np.load('Target_1.npy', allow_pickle=True)
        Global_Vars.Feat = Images
        Global_Vars.Tar = Target
        Npop = 10
        Chlen = 6  # hidden neuron count ,Epochs and No of dense block in Densenet , hidden neuron count ,Epochs and No of resblock in Resnet
        xmin = np.asarray([5, 5, 3, 5, 5, 3]) * np.ones((Npop, 1))
        xmax = np.asarray([255, 50, 15, 255, 50, 15]) * np.ones((Npop, 1))
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
        fname = objfun_cls
        max_iter = 50

        print('PSO....')
        [bestfit1, fitness1, bestsol1, Time1] = PSO(initsol, fname, xmin, xmax, max_iter)

        print('JAYA....')
        [bestfit2, fitness2, bestsol2, Time2] = JAYA(initsol, fname, xmin, xmax, max_iter)

        print('GSO....')
        [bestfit3, fitness3, bestsol3, Time3] = GSO(initsol, fname, xmin, xmax, max_iter)

        print('ROA....')
        [bestfit4, fitness4, bestsol4, Time4] = ROA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

        sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]

        Best_sol.append(sol)
    np.save('Best_cls_sol.npy', Best_sol)

# CLASSIFICATION
an = 0
if an == 1:
    Eval_all = []
    for n in range(no_of_dataset):
        Feat = np.load('Segmentation_1.npy', allow_pickle=True)
        Target = np.load('Target_1.npy', allow_pickle=True)
        best = np.load('Best_cls_sol.npy', allow_pickle=True)
        Target = np.reshape(Target, (-1, 1))
        EVAL = []
        Learnper = [0.35, 0.55, 0.65, 0.75, 0.85]
        for learn in range(len(Learnper)):
            learnperc = round(Feat.shape[0] * Learnper[learn])
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 14))
            train_data = np.reshape(Train_Data, (Train_Data.shape[0], Train_Data.shape[1] * Train_Data.shape[2]))
            train_data = np.resize(train_data, (train_data.shape[0], 100))
            test_data = np.reshape(Test_Data, (Test_Data.shape[0], Test_Data.shape[1] * Test_Data.shape[2]))
            test_data = np.resize(test_data, (test_data.shape[0], 100))
            for j in range(best.shape[0]):
                sol = np.round((best[n][learn][j])).astype(int)
                Eval[j, :], pred1 = model_proposed(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval[5, :], pred2 = Model_CNN(train_data, Train_Target, test_data, Test_Target)
            Eval[6, :], pred3 = Model_VGG16(train_data, Train_Target, test_data, Test_Target)
            Eval[7, :], pred4 = Model_RESNET(train_data, Train_Target, test_data, Test_Target)
            Eval[8, :], pred5 = model_proposed(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval[9, :], pred6 = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Eval_all.npy', Eval_all)



plot_results()
plot_results_table()
plot_confusion()
plot_results_conv()
Plot_ROC()
Image_Results_seg()
