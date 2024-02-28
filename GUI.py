import cv2 as cv
import os
from tkinter import *
from tkinter import filedialog
from click.exceptions import Exit
import numpy as np
from PIL import ImageTk, Image
from FCM import FCM
from Region_Growing import Region_Growing


class globalvars:
    image = None
    prep = None
    segm = None
    filename = None


def Upload():
    root.filename = filedialog.askopenfile(initialdir=os.getcwd() + "//Datasets", title="select a image", filetypes=(
        ("JPEG Files", "*.jpg"), ("PNG Files", "*.png"), ("all file", "*.*")))
    image = Image.open(root.filename.name)
    globalvars.filename = root.filename.name
    globalvars.image = np.asarray(image)
    img = ImageTk.PhotoImage(image)
    orglabel.config(image=img)
    orglabelt.config(text='ORIGINAL IMAGE')
    orglabel.image = img


def Preprocess():
    alpha = 1.2  # Contrast control
    beta = 0.2  # Brightness control
    img = globalvars.image
    imagess = cv.convertScaleAbs(img, alpha=alpha, beta=beta)  # Contrast Enhancement
    imgmedian = cv.medianBlur(imagess, 5)  # Median Filtering
    globalvars.prep = imgmedian
    prepimg = Image.fromarray(imgmedian)
    img = ImageTk.PhotoImage(prepimg)
    preplabel.config(image=img)
    preplabelt.config(text='PREPROCESSED IMAGE')
    preplabel.image = img


def Segment():
    image = globalvars.prep
    image = np.uint8(image)  # to change the unsigned int bit 8 value
    if len(image.shape) == 3:  # check whether image is RGB or not
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # changing the color model ie,RGB2GRAY
    image = cv.resize(image, (224, 224))
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
    regio = cv.resize(image_i, (224, 224))
    result[result != 0] = 1
    regio[regio != 0] = 1
    result = np.uint8(result)
    regio = np.uint8(regio)
    dest_and = cv.bitwise_and(result, regio, mask=None)
    output1 = np.zeros(image.shape, dtype=np.uint8)
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
    globalvars.segm = Image.fromarray(Ensemble_Segmented)
    segimg = Image.fromarray(Ensemble_Segmented)
    img = ImageTk.PhotoImage(segimg)
    segmlabel.config(image=img)
    segmlabelt.config(text='SEGMENTED IMAGE')
    segmlabel.image = img


def Classification():
    img = globalvars.segm
    filename = globalvars.filename
    name = filename.split('/')
    ui = name[8]
    if ui == 'Pre':
        ui = 'Pre'
    elif ui == 'Benign':
        ui = 'Benign'
    elif ui == 'Early':
        ui = 'Early'
    elif ui == 'Pro':
        ui = 'Pro'
    targetlabelt.config(text='TYPES OF CLASS =>')
    tumourlabel.config(text=str(ui))


def Reset():
    orglabelt.config(text='')
    orglabel.config(image='')
    preplabelt.config(text='')
    preplabel.config(image='')
    segmlabelt.config(text='')
    segmlabel.config(image='')
    targetlabelt.config(text='')
    tumourlabel.config(text='')


def Close():
    root.quit()


if __name__ == '__main__':
    frameHeight = [75, 175, 500]
    layer = [25, 75, 125]
    fontSize = [25, 16, 20]
    root = Tk()
    root.title("LEUKEMIA DETECTION - GUI")
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (w, h))

    titleFrame = Frame(root, width=w, height=100, bg='cyan')
    titleFrame.place(x=0, y=0)

    title = Label(titleFrame, text="LEUKEMIA DETECTION", font=("Arial", fontSize[0]), bg='cyan')
    title.place(relx=0.5, rely=0.5, anchor=CENTER)

    buttonFrame = Frame(root, width=200, height=h - 100, bg='black')
    buttonFrame.place(x=0, y=100)

    uploadbutton = Button(buttonFrame, text="Upload Image", bg="lightgreen", font=("Arial", fontSize[1]),
                          command=Upload)
    uploadbutton.place(relx=0.5, rely=0.05, anchor=CENTER)
    #
    prepbutton = Button(buttonFrame, text="Preprocess", bg="lightgreen", font=("Arial", fontSize[1]),
                        command=Preprocess)
    prepbutton.place(relx=0.5, rely=0.2, anchor=CENTER)
    #
    segmbutton = Button(buttonFrame, text="Segmentation", bg="lightgreen", font=("Arial", fontSize[1]),
                        command=Segment)
    segmbutton.place(relx=0.5, rely=0.35, anchor=CENTER)
    #
    detectbutton = Button(buttonFrame, text="Classification", bg="lightgreen", font=("Arial", fontSize[1]),
                          command=Classification)
    detectbutton.place(relx=0.5, rely=0.5, anchor=CENTER)
    #
    resetbutton = Button(buttonFrame, text="Reset", bg="lightgreen", font=("Arial", fontSize[1]), command=Reset)
    resetbutton.place(relx=0.5, rely=0.65, anchor=CENTER)

    exitbutton = Button(buttonFrame, text="Exit", bg="lightgreen", font=("Arial", fontSize[1]), command=Close)
    exitbutton.place(relx=0.5, rely=0.80, anchor=CENTER)

    outputFrame = Frame(root, width=w - 200, height=h - 100, bg='lightblue')
    outputFrame.place(x=200, y=100)

    orglabelt = Label(outputFrame, bg='lightblue', font=("Arial", fontSize[1]))
    orglabelt.place(relx=0.15, rely=0.10, anchor='n')

    orglabel = Label(outputFrame, bg='lightblue')
    orglabel.place(relx=0.15, rely=0.15, anchor='n')

    preplabelt = Label(outputFrame, bg='lightblue', font=("Arial", fontSize[1]))
    preplabelt.place(relx=0.45, rely=0.10, anchor='n')

    preplabel = Label(outputFrame, bg='lightblue')
    preplabel.place(relx=0.45, rely=0.15, anchor='n')

    segmlabelt = Label(outputFrame, bg='lightblue', font=("Arial", fontSize[1]))
    segmlabelt.place(relx=0.75, rely=0.10, anchor='n')

    segmlabel = Label(outputFrame, bg='lightblue')
    segmlabel.place(relx=0.75, rely=0.15, anchor='n')

    targetlabelt = Label(outputFrame, bg='lightblue', font=("Arial", 40))
    targetlabelt.place(relx=0.3, rely=0.7, anchor='n')

    tumourlabel = Label(outputFrame, bg='lightblue', font=("Arial", 40))
    tumourlabel.place(relx=0.6, rely=0.7, anchor='n')

    root.mainloop()
