import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def Image_Results_seg():
    Images = np.load('Images_1.npy', allow_pickle=True)
    preproces = np.load('Preprocess_1.npy', allow_pickle=True)
    GT = np.load('Ground_Truth_1.npy', allow_pickle=True)
    rg = np.load('segment_1_region.npy', allow_pickle=True)
    kms = np.load('segment_1_kms.npy', allow_pickle=True)
    fcm = np.load('segment_1_fcm.npy', allow_pickle=True)
    Segmented = np.load('Segmentation_1.npy', allow_pickle=True)
    ind = [1, 3, 5, 7, 10]  # 8,9,11,32,50
    for i in range(len(ind)):
        image = Images[ind[i]]  # Images[Image[n][j]]
        prepro = preproces[ind[i]]
        Ground_Truth = GT[ind[i]]
        RG = rg[ind[i]]
        KMS = kms[ind[i]]
        FCM = fcm[ind[i]]
        Seg = Segmented[ind[i]]
        plt.suptitle("Image Result %d" % (i + 1), fontsize=20)
        plt.subplot(2, 4, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.subplot(2, 4, 2)
        plt.title('preproces')
        plt.imshow(prepro)
        plt.subplot(2, 4, 3)
        plt.title('Ground Truth')
        plt.imshow(Ground_Truth)
        plt.subplot(2, 4, 4)
        plt.title('Region Growing')
        plt.imshow(RG)
        plt.subplot(2, 4, 5)
        plt.title('K-Medoids')
        plt.imshow(KMS)
        plt.subplot(2, 4, 6)
        plt.title('FCM')
        plt.imshow(FCM)
        plt.subplot(2, 4, 7)
        plt.title('Segmented Image')
        plt.imshow(Seg)
        plt1 = plt.subplot(2, 4, 8)
        plt1.axis('off')
        path1 = "./Results/Dataset_%simage.png" % (i + 1)
        plt.savefig(path1)
        plt.show()
        cv.imwrite('./Results/Img_Res/original-' + str(i + 1) + '.png', image)
        cv.imwrite('./Results/Img_Res/preprocess-' + str(i + 1) + '.png', prepro)
        cv.imwrite('./Results/Img_Res/Ground_Truth-' + str(i + 1) + '.png', Ground_Truth)
        cv.imwrite('./Results/Img_Res/Region Growing-' + str(i + 1) + '.png', RG)
        cv.imwrite('./Results/Img_Res/K-Medoids-' + str(i + 1) + '.png', KMS)
        cv.imwrite('./Results/Img_Res/FCM-' + str(i + 1) + '.png', FCM)
        cv.imwrite('./Results/Img_Res/segmented-' + str(i + 1) + '.png', Seg)


if __name__ == '__main__':
    Image_Results_seg() # for segmentation
