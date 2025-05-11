import os

import cv2
import pandas as pd
import torch

from img2graphOrigin import imgToNXGraph
from sklearn.model_selection import train_test_split


def saveDocumentGra(imgsFirstPath, saveFirstPath):
    imgNameList = os.listdir(imgsFirstPath)
    # 查看数量
    nums = len(imgNameList)
    imgTypes = ['.tiff', '.png', '.jpeg', '.jpg']
    imgType = imgNameList[int(nums / 2)].split(".")[1]
    # data size
    size = 5
    begin = int((nums-size) / 2)

    for n in imgNameList[begin-1 : begin-1+size]:
        # if(n.split(".")[1] not in imgTypes):
        if (n.split(".")[1] != imgType):
            continue

        imgPath = os.path.join(imgsFirstPath, n)
        saveN = n.split(".")[0] + ".pt"
        savePath = os.path.join(saveFirstPath, saveN)
        # 获取图数据
        graph = imgToNXGraph(imgPath)

        torch.save(graph, savePath)
def saveDocumentImg(imgsFirstPath, saveFirstPath):
    imgNameList = os.listdir(imgsFirstPath)
    # 查看数量
    nums = len(imgNameList)
    imgTypes = ['.tiff', '.png', '.jpeg', '.jpg']
    imgType = imgNameList[int(nums / 2)].split(".")[1]
    # data size
    size = 5
    begin = int((nums-size) / 2)

    for n in imgNameList[begin-1 : begin-1+size]:
        # if(n.split(".")[1] not in imgTypes):
        if (n.split(".")[1] != imgType):
            continue

        imgPath = os.path.join(imgsFirstPath, n)
        saveN = n
        savePath = os.path.join(saveFirstPath, saveN)
        # 保存图像数据
        img = cv2.imread(imgPath)
        cv2.imwrite(savePath, img)

def RAFDBDataGraph():
    path = "E:\\document\\python\\MyDriveData\\RAF-DB\\basic\\EmoLabel\\list_patition_label.txt"

    # 打开文件
    with open(path, 'r') as file:
        train = []
        test = []
        for line in file:
            # 去除行末的换行符和空白字符
            line = line.strip()
            # 将每行数据按逗号分隔并转换为整数
            tem = line.split(" ")
            if("train" in tem[0]):
                train.append([tem[0], int(tem[1])])
            else:
                test.append([tem[0], int(tem[1])])
    train = pd.DataFrame(train, columns=["name", "label"])
    test = pd.DataFrame(test, columns=["name", "label"])
    # 设置保存的地方
    imgsFirstPath = "E:\\document\\python\\MyDriveData\\RAF-DB\\basic\\Image\\aligned"
    saveFirstPath = "E:\\document\\python\\MyDriveData\\RAF-DB\\basic\\ImgGraph"
    # 创建文件
    if os.path.exists(saveFirstPath) == False:
        os.makedirs(saveFirstPath)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    # names = test["name"].tolist()
    # trSize = int(len(names)*0.8)
    # vaSize = int(len(names)*0.1)
    # teSize = len(names)-trSize-vaSize
    for i, data in enumerate([test, val, train]):
        print(f"processing mode{i}")
        if(i == 0):
            m = "test"
        elif(i==1):
            m = "val"
        else:
            m = "train"
        saveSecondPath = os.path.join(saveFirstPath, m)
        data.to_csv(os.path.join(saveFirstPath, f"{m}Y.csv"), index=False)
        # 创建文件
        if os.path.exists(saveSecondPath) == False:
            os.makedirs(saveSecondPath)

        for j, n in enumerate(data['name'].tolist()):
            imgPath = os.path.join(imgsFirstPath, n.split(".")[0] + "_aligned." + n.split(".")[1])
            saveN = n.split(".")[0] + ".pt"
            savePath = os.path.join(saveSecondPath, saveN)
            # 获取图数据
            center, graph = imgToNXGraph(imgPath, rW=48, rH=48)

            torch.save(graph, savePath)

def Fer2013DataGraph(originStage=True):
    if(originStage == True):
        # path = r"E:\document\python\MyDriveData\FER2013-FERPlus\FERPlus\Image\alignedYLabel.csv"
        path = r"../databases/fer2013/images/alignedYLabel.csv"
        total = pd.read_csv(path)
        total['flag'] = total['name'].map(lambda x: 1 if "train" in x else 0)
        train = total[total['flag']==1]
        test = total[total['flag']==0]
        train = train[["name", "label"]]
        test = test[["name", "label"]]
        # divideLabel = train['label'].tolist()
        # train, val, trY, vaY = train_test_split(train, divideLabel, test_size=0.1, random_state=42, stratify=divideLabel)
        train, val = train_test_split(train, test_size=0.1, random_state=42)
    else:
        path = r"../databases/fer2013/graph/granuarRaw"
        train = pd.read_csv(os.path.join(path, "trainY.csv"))
        val = pd.read_csv(os.path.join(path, "valY.csv"))
        test = pd.read_csv(os.path.join(path, "testY.csv"))
    # 设置保存的地方
    # imgsFirstPath = "D:\\Documents\\Python\\myDriveData\\fer2013\\images\\aligned"
    imgsFirstPath = r"../databases/fer2013/images/aligned"
    saveFirstPath = r"../databases/fer2013/graph/granuarRaw"
    # 创建文件
    if os.path.exists(saveFirstPath) == False:
        os.makedirs(saveFirstPath)

    for i, data in enumerate([test, val, train]):
        print(f"processing mode{i}")
        if(i == 0):
            m = "test"
        elif(i==1):
            m = "val"
        else:
            m = "train"
        saveSecondPath = os.path.join(saveFirstPath, m)
        data.to_csv(os.path.join(saveFirstPath, f"{m}Y.csv"), index=False)
        # 创建文件
        if os.path.exists(saveSecondPath) == False:
            os.makedirs(saveSecondPath)

        for j, n in enumerate(data['name'].tolist()):
            imgPath = os.path.join(imgsFirstPath, n)
            saveN = n.split(".")[0] + ".pt"
            savePath = os.path.join(saveSecondPath, saveN)
            # 获取图数据
            center, graph = imgToNXGraph(imgPath, rW=48, rH=48)

            torch.save(graph, savePath)

def AffectNetDataGraph():
    rW = rH = 224
    trPath = r"E:\document\python\MyDriveData\AffectNet\images\train-sample-affectnet.csv"
    tePath = r"E:\document\python\MyDriveData\AffectNet\images\valid-sample-affectnet.csv"
    train = pd.read_csv(trPath)
    test = pd.read_csv(tePath)
    train['label'] = train['emotion'].tolist()
    train['name'] = train['image'].map(lambda x: "__".join(x.split("/")[-3:]))
    test['label'] = test['emotion'].tolist()
    test['name'] = test['image'].map(lambda x: "__".join(x.split("/")[-3:]))
    train = train[["name", "label"]]
    test = test[["name", "label"]]
    # 设置保存的地方
    imgsFirstPath = "E:\\document\\python\\MyDriveData\\AffectNet\\images"
    saveFirstPath = "E:\\document\\python\\MyDriveData\\AffectNet\\ImgGraph"
    # 创建文件
    if os.path.exists(saveFirstPath) == False:
        os.makedirs(saveFirstPath)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    for i, data in enumerate([test, val, train]):
        print(f"processing mode{i}")
        if(i == 0):
            m = "test"
        elif(i==1):
            m = "val"
        else:
            m = "train"
        saveSecondPath = os.path.join(saveFirstPath, m)
        data.to_csv(os.path.join(saveFirstPath, f"{m}Y.csv"), index=False)
        # 创建文件
        if os.path.exists(saveSecondPath) == False:
            os.makedirs(saveSecondPath)

        for j, n in enumerate(data['name'].tolist()):
            imgPath = os.path.join(imgsFirstPath, "/".join(n.split("__")))
            saveN = n.split(".")[0] + ".pt"
            savePath = os.path.join(saveSecondPath, saveN)
            # 获取图数据
            center, graph = imgToNXGraph(imgPath, rW=rW, rH=rH)

            torch.save(graph, savePath)

def Affect7DataGraph():
    basePath = r"D:\Documents\Python\myDriveData\AffectNet_kaggle\images\train"
    emotions = sorted([f for f in os.listdir(basePath) if
                       not f.startswith('.')], key=lambda f: f.lower())
    print(f"emotion sorted: {emotions}")
    dataNames = []
    labels = []
    for l, eN in enumerate(emotions):
        dataSecPath = os.path.join(basePath, eN)
        dataList = sorted(["train__" + eN + "__" + f for f in os.listdir(dataSecPath) if
                           not f.startswith('.')], key=lambda f: f.lower())
        labelList = [l] * len(dataList)
        dataNames.extend(dataList)
        labels.extend(labelList)
    train = pd.DataFrame({"name": dataNames,
                         "label": labels})
    basePath = r"D:\Documents\Python\myDriveData\AffectNet_kaggle\images\val"
    dataNames = []
    labels = []
    for l, eN in enumerate(emotions):
        dataSecPath = os.path.join(basePath, eN)
        dataList = sorted(["val__" + eN + "__" + f for f in os.listdir(dataSecPath) if
                           not f.startswith('.')], key=lambda f: f.lower())
        labelList = [l] * len(dataList)
        dataNames.extend(dataList)
        labels.extend(labelList)
    val = pd.DataFrame({"name": dataNames,
                          "label": labels})
    basePath = r"D:\Documents\Python\myDriveData\AffectNet_kaggle\images\test"
    dataNames = []
    labels = []
    for l, eN in enumerate(emotions):
        dataSecPath = os.path.join(basePath, eN)
        dataList = sorted(["test__" + eN + "__" + f for f in os.listdir(dataSecPath) if
                           not f.startswith('.')], key=lambda f: f.lower())
        labelList = [l] * len(dataList)
        dataNames.extend(dataList)
        labels.extend(labelList)
    test = pd.DataFrame({"name": dataNames,
                        "label": labels})
    # again divide
    dataNames = train['name'].tolist()+val['name'].tolist()+test['name'].tolist()
    labels = train['label'].tolist()+val['label'].tolist()+test['label'].tolist()
    trainX, vateX, trainY, vateY = train_test_split(dataNames, labels, test_size=0.2,
                                                  random_state=42, stratify=labels)
    valX, testX, valY, testY = train_test_split(vateX, vateY, test_size=0.5,
                                                    random_state=42, stratify=vateY)
    train = pd.DataFrame({"name": trainX,
                         "label": trainY})
    val = pd.DataFrame({"name": valX,
                         "label": valY})
    test = pd.DataFrame({"name": testX,
                         "label": testY})
    # 设置保存的地方
    imgsFirstPath = "D:\\Documents\\Python\\myDriveData\\AffectNet_kaggle\\images"
    saveFirstPath = "D:\\Documents\\Python\\myDriveData\\AffectNet_kaggle\\ImgGraph"
    # 创建文件
    if os.path.exists(saveFirstPath) == False:
        os.makedirs(saveFirstPath)

    for i, data in enumerate([test, val, train]):
        print(f"processing mode{i}")
        if(i == 0):
            m = "test"
        elif(i==1):
            m = "val"
        else:
            m = "train"
        saveSecondPath = os.path.join(saveFirstPath, m)
        data.to_csv(os.path.join(saveFirstPath, f"{m}Y.csv"), index=False)
        # 创建文件
        if os.path.exists(saveSecondPath) == False:
            os.makedirs(saveSecondPath)

        # for j, n in enumerate(data['name'].tolist()):
        #     imgPath = os.path.join(imgsFirstPath, "/".join(n.split("__")))
        #     saveN = n.split(".")[0] + ".pt"
        #     savePath = os.path.join(saveSecondPath, saveN)
        #     # 获取图数据
        #     center, graph = imgToNXGraph(imgPath, rW=rW, rH=rH)
        #
        #     torch.save(graph, savePath)

def SFEWDataGraph():
    img = cv2.imread(r"D:\Documents\Python\myDriveData\SFEW 2.0\images\Train_Aligned_Faces\Angry\Airheads_000519240_00000005.png", cv2.IMREAD_GRAYSCALE)
    (rH, rW) = img.shape
    # rW = rH = 224
    basePath = r"D:\Documents\Python\myDriveData\SFEW 2.0\images\Train_Aligned_Faces"
    emotions = sorted([f for f in os.listdir(basePath) if
                       not f.startswith('.')], key=lambda f: f.lower())
    print(f"emotion sorted: {emotions}")
    dataNames = []
    labels = []
    for l, eN in enumerate(emotions):
        dataSecPath = os.path.join(basePath, eN)
        dataList = sorted(["Train_Aligned_Faces__" + eN + "__" + f for f in os.listdir(dataSecPath) if
                           not f.startswith('.')], key=lambda f: f.lower())
        labelList = [l] * len(dataList)
        dataNames.extend(dataList)
        labels.extend(labelList)
    trainX, valX, trainY, valY = train_test_split(dataNames, labels, test_size=0.1,
                                                    random_state=42, stratify=labels)
    train = pd.DataFrame({"name": trainX,
                          "label": trainY})
    val = pd.DataFrame({"name": valX,
                        "label": valY})
    basePath = r"D:\Documents\Python\myDriveData\SFEW 2.0\images\Val_Aligned_Faces"
    dataNames = []
    labels = []
    for l, eN in enumerate(emotions):
        dataSecPath = os.path.join(basePath, eN)
        dataList = sorted(["Val_Aligned_Faces__" + eN + "__" + f for f in os.listdir(dataSecPath) if
                           not f.startswith('.')], key=lambda f: f.lower())
        labelList = [l] * len(dataList)
        dataNames.extend(dataList)
        labels.extend(labelList)
    test = pd.DataFrame({"name": dataNames,
                          "label": labels})

    # 设置保存的地方
    imgsFirstPath = "D:\\Documents\\Python\\myDriveData\\SFEW 2.0\\images"
    saveFirstPath = "D:\\Documents\\Python\\myDriveData\\SFEW 2.0\\ImgGraph"
    # 创建文件
    if os.path.exists(saveFirstPath) == False:
        os.makedirs(saveFirstPath)

    for i, data in enumerate([test, val, train]):
        print(f"processing mode{i}")
        if(i == 0):
            m = "test"
        elif(i==1):
            m = "val"
        else:
            m = "train"
        saveSecondPath = os.path.join(saveFirstPath, m)
        data.to_csv(os.path.join(saveFirstPath, f"{m}Y.csv"), index=False)
        # 创建文件
        if os.path.exists(saveSecondPath) == False:
            os.makedirs(saveSecondPath)

        for j, n in enumerate(data['name'].tolist()):
            imgPath = os.path.join(imgsFirstPath, "/".join(n.split("__")))
            saveN = n.split(".")[0] + ".pt"
            savePath = os.path.join(saveSecondPath, saveN)
            # 获取图数据
            center, graph = imgToNXGraph(imgPath, rW=rW, rH=rH)

            torch.save(graph, savePath)

def CAER_SDataGraph():
    # img = cv2.imread(r"D:\Documents\Python\myDriveData\SFEW 2.0\images\Train_Aligned_Faces\Angry\Airheads_000519240_00000005.png", cv2.IMREAD_GRAYSCALE)
    # (rH, rW) = img.shape
    rW = rH = 224
    basePath = r"../databases/CAER-S/images/train"
    emotions = sorted([f for f in os.listdir(basePath) if
                       not f.startswith('.')], key=lambda f: f.lower())
    print(f"emotion sorted: {emotions}")
    dataNames = []
    labels = []
    for l, eN in enumerate(emotions):
        dataSecPath = os.path.join(basePath, eN)
        dataList = sorted(["train__" + eN + "__" + f for f in os.listdir(dataSecPath) if
                           not f.startswith('.')], key=lambda f: f.lower())
        labelList = [l] * len(dataList)
        dataNames.extend(dataList)
        labels.extend(labelList)
    trainX, valX, trainY, valY = train_test_split(dataNames, labels, test_size=0.1,
                                                    random_state=42, stratify=labels)
    train = pd.DataFrame({"name": trainX,
                          "label": trainY})
    val = pd.DataFrame({"name": valX,
                        "label": valY})
    basePath = r"../databases/CAER-S/images/test"
    dataNames = []
    labels = []
    for l, eN in enumerate(emotions):
        dataSecPath = os.path.join(basePath, eN)
        dataList = sorted(["test__" + eN + "__" + f for f in os.listdir(dataSecPath) if
                           not f.startswith('.')], key=lambda f: f.lower())
        labelList = [l] * len(dataList)
        dataNames.extend(dataList)
        labels.extend(labelList)
    test = pd.DataFrame({"name": dataNames,
                          "label": labels})

    # 设置保存的地方
    imgsFirstPath = r"../databases/CAER-S/images"
    saveFirstPath = r"../databases/CAER-S/graph/granuarRaw"
    # 创建文件
    if os.path.exists(saveFirstPath) == False:
        os.makedirs(saveFirstPath)

    for i, data in enumerate([test, val, train]):
        print(f"processing mode{i}")
        if(i == 0):
            m = "test"
        elif(i==1):
            m = "val"
        else:
            m = "train"
        saveSecondPath = os.path.join(saveFirstPath, m)
        data.to_csv(os.path.join(saveFirstPath, f"{m}Y.csv"), index=False)
        # 创建文件
        if os.path.exists(saveSecondPath) == False:
            os.makedirs(saveSecondPath)

        for j, n in enumerate(data['name'].tolist()):
            imgPath = os.path.join(imgsFirstPath, "/".join(n.split("__")))
            saveN = n.split(".")[0] + ".pt"
            savePath = os.path.join(saveSecondPath, saveN)
            # 获取图数据
            center, graph = imgToNXGraph(imgPath, rW=rW, rH=rH)

            torch.save(graph, savePath)

def CKDataGraph():
    rW, rH = 48, 48
    basePath = "E:\\document\\python\\MyDriveData\\CK+48\\images"
    emotions = sorted([f for f in os.listdir(basePath) if
                        not f.startswith('.')], key=lambda f: f.lower())
    print(f"emotion sorted: {emotions}")
    dataNames = []
    labels = []
    for l, eN in enumerate(emotions):
        dataSecPath = os.path.join(basePath, eN)
        dataList = sorted([eN+"__"+f for f in os.listdir(dataSecPath) if
                        not f.startswith('.')], key=lambda f: f.lower())
        labelList = [l] * len(dataList)
        dataNames.extend(dataList)
        labels.extend(labelList)

    trainX, vateX, trainY, vateY = train_test_split(dataNames, labels, test_size=0.2,
                                                    random_state=42, stratify=labels)
    valX, testX, valY, testY = train_test_split(vateX, vateY, test_size=0.5,
                                                    random_state=42, stratify=vateY)
    train = pd.DataFrame({"name": trainX,
                          "label": trainY})
    val = pd.DataFrame({"name": valX,
                          "label": valY})
    test = pd.DataFrame({"name": testX,
                          "label": testY})

    imgsFirstPath = "E:\\document\\python\\MyDriveData\\CK+48\\images"
    saveFirstPath = "E:\\document\\python\\MyDriveData\\CK+48\\ImgGraph"
    # 创建文件
    if os.path.exists(saveFirstPath) == False:
        os.makedirs(saveFirstPath)

    for i, data in enumerate([train, val, test]):
        print(f"processing mode{i}")
        if(i == 0):
            m = "train"
        elif(i==1):
            m = "val"
        else:
            m = "test"
        saveSecondPath = os.path.join(saveFirstPath, m)
        data.to_csv(os.path.join(saveFirstPath, f"{m}Y.csv"), index=False)
        # 创建文件
        if os.path.exists(saveSecondPath) == False:
            os.makedirs(saveSecondPath)

        for j, n in enumerate(data['name'].tolist()):
            imgPath = os.path.join(imgsFirstPath, n.split("__")[0] + "\\" + n.split("__")[1])
            saveN = n.split(".")[0] + ".pt"
            savePath = os.path.join(saveSecondPath, saveN)
            # 获取图数据
            center, graph = imgToNXGraph(imgPath, rW=rW, rH=rH)

            torch.save(graph, savePath)

def OuluDataProcess():
    saveOriginPath = r"D:\Documents\Python\myDriveData\Oulu_CASIA_NIR_VIS\images"
    baseOriPath = r"D:\Documents\Python\myDriveData\Oulu_CASIA_NIR_VIS"
    for pty in ['NI', "VL"]:
        basePath = os.path.join(baseOriPath, pty)
        types = os.listdir(basePath)
        for ty in types:
            print(f"processing {ty}")
            firstDataPath = os.path.join(basePath, ty)
            samples = os.listdir(firstDataPath)
            for sa in samples:
                secondDataPath = os.path.join(firstDataPath, sa)
                classes = os.listdir(secondDataPath)
                for cla in classes:
                    thirdDataPath = os.path.join(secondDataPath, cla)
                    savePath = os.path.join(saveOriginPath, cla)
                    if (os.path.exists(savePath) == False):
                        os.makedirs(savePath)
                    imgList = os.listdir(thirdDataPath)
                    for iN in imgList:
                        print(iN)
                        ext = None
                        if("." in iN):
                            ext = iN.split(".")[1]
                        else:
                            continue
                        if ext not in ["jpg", "jpeg", "png", "tiff"]:
                            continue
                        dataPath = os.path.join(thirdDataPath, iN)
                        # print(dataPath)
                        img = cv2.imread(dataPath)
                        saveDataPath = os.path.join(savePath, f"{pty}_{ty}_{sa}_{cla}_{iN}")
                        print(saveDataPath)
                        cv2.imwrite(saveDataPath, img)

def SAMMDataGraph():
    basePath = r"D:\Documents\Python\myDriveData\SAMM\images"
    samples = sorted([f for f in os.listdir(basePath) if
                       not f.startswith('.')], key=lambda f: f.lower())
    oriLabel = pd.read_excel(r"D:\Documents\Python\myDriveData\SAMM\SAMM_Micro_FACS_Codes_v2.xlsx", header=13)
    # labels = oriLabel['Objective Classes'].tolist()
    # trainXOri, vateX, trainYOri, vateY = train_test_split(oriLabel, labels, test_size=0.2,
    #                                                 random_state=42, stratify=labels)
    # valXOri, testXOri, valYOri, testYOri = train_test_split(vateX, vateY, test_size=0.5,
    #                                             random_state=42, stratify=vateY)
    # trainX = trainY = []
    # valX = valY = []
    # testX = testY = []
    dataNames = []
    labels = []
    emotion = list(set(oriLabel['Estimated Emotion'].values))
    newEmotion = []
    for n in emotion:
        if(n == "Other"):
            continue
        newEmotion.append(n)
    emoDict = dict()
    for l,n in enumerate(newEmotion):
        emoDict[n] = l
    for l, sa in enumerate(samples):
        dataSecPath = os.path.join(basePath, sa)
        frames = os.listdir(dataSecPath)
        for i, fr in enumerate(frames):
            dataThirdPath = os.path.join(dataSecPath, fr)
            label = list(oriLabel[oriLabel['Filename'] == fr]['Estimated Emotion'].values)[0]
            if(label == "Other"):
                continue
            dataList = sorted([sa + "__" + fr + "__" + f for f in os.listdir(dataThirdPath) if
                               not f.startswith('.')], key=lambda f: f.lower())
            labelList = [emoDict[label]] * len(dataList)
            dataNames.extend(dataList)
            labels.extend(labelList)

    trainX, vateX, trainY, vateY = train_test_split(dataNames, labels, test_size=0.6,
                                                  random_state=42, stratify=labels)
    valX, testX, valY, testY = train_test_split(vateX, vateY, test_size=0.83,
                                                    random_state=42, stratify=vateY)
    train = pd.DataFrame({"name": trainX,
                         "label": trainY})
    val = pd.DataFrame({"name": valX,
                         "label": valY})
    test = pd.DataFrame({"name": testX,
                         "label": testY})
    # 设置保存的地方
    imgsFirstPath = "D:\\Documents\\Python\\myDriveData\\SAMM\\images"
    saveFirstPath = "D:\\Documents\\Python\\myDriveData\\SAMM\\ImgGraph"
    # 创建文件
    if os.path.exists(saveFirstPath) == False:
        os.makedirs(saveFirstPath)

    for i, data in enumerate([test, val, train]):
        print(f"processing mode{i}")
        if(i == 0):
            m = "test"
        elif(i==1):
            m = "val"
        else:
            m = "train"
        saveSecondPath = os.path.join(saveFirstPath, m)
        data.to_csv(os.path.join(saveFirstPath, f"{m}Y.csv"), index=False)
        # 创建文件
        if os.path.exists(saveSecondPath) == False:
            os.makedirs(saveSecondPath)

        # for j, n in enumerate(data['name'].tolist()):
        #     imgPath = os.path.join(imgsFirstPath, "/".join(n.split("__")))
        #     saveN = n.split(".")[0] + ".pt"
        #     savePath = os.path.join(saveSecondPath, saveN)
        #     # 获取图数据
        #     center, graph = imgToNXGraph(imgPath, rW=rW, rH=rH)
        #
        #     torch.save(graph, savePath)

if __name__ == "__main__":
    # RAFDBDataGraph()
    # CKDataGraph()
    # Fer2013DataGraph(originStage=False)
    # AffectNetDataGraph()
    # SFEWDataGraph()
    CAER_SDataGraph()
    # Affect7DataGraph()
    # OuluDataProcess()
    # OuluDataGraph()
    # SAMMDataGraph()