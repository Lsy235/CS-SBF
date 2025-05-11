import cv2
import numpy as np
import time
import networkx as nx
import scipy.sparse as sp
import random
random.seed(0)

# 返回所有像素梯度信息
def get_grad(img, dir = None):
    """
    :param img:
    :return: map{(x,y):grad} and grad matric
    """
    # get grad list
    # 分别计算x、y方向：右减左，下减上
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy2 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)  # 梯度矩阵 : same shape with the img

    return sobelxy2

def cal_bound(img, center, Rx, Ry):
    """
    :param img: 原图，需要获取原图的大小
    :param center: 中心点坐标
    :param Rx, Ry: 不同方向的半径
    :return: 上下左右边界
    """
    h, w = img.shape
    left = center[1] - Rx
    right = center[1] + Rx
    up = center[0] - Ry
    down = center[0] + Ry

    if left < 0 :
        left = 0
    if right >= w :
        right = w - 1
    if up < 0 :
        up = 0
    if down >= h :
        down = h - 1

    return int(left), int(right), int(up), int(down)

def center_select(img_grad, img_label):  # 获得梯度最小的点
    """
    :param img_grad:
    :param img_label:
    :return:
    """
    minPix_xy = np.where(img_grad == img_grad.min()) # 返回一个二维元组，前一个为所有梯度最小的点的横坐标，后一个为所有梯度最小的点的纵坐标

    while True:
        pos = random.randint(0, len(minPix_xy[0]) - 1)  # 随机挑选一个中心点
        #print(len(minPix_xy[0]))
        if (img_label[minPix_xy[0][pos],minPix_xy[1][pos]] != 1):
            return [minPix_xy[0][pos],minPix_xy[1][pos]]

def cal_Radius(img, center, purity, threshold, var_threshold):
    """
    :param img: 输入图片
    :param center:  中心点坐标 [x, y]
    :param purity:  1 - （异类点个数 / 总个数)
    :param threshold:  判断是否是异类点， 与中心点灰度值的差值的绝对值 / 中心点的灰度值
    :return: Rx, Ry, 输入中心点对应的半径
    该方法待优化
    """
    Rx = 0 # 初始化半径
    Ry = 0

    flag = True
    flag_x = True
    flag_y = True
    item_count = 0
    temp_pixNum = 0
    center_value = int(img[center[0], center[1]])

    while True:
        if flag_x == True and flag_y == True:
            item_count += 1
        else:
            if flag_x:
                item_count = 1
            if flag_y:
                item_count = 2
        if flag_x and item_count % 2 != 0:
            Rx += 1

        if flag_y and item_count % 2 == 0:
            Ry += 1

        #计算切片边界
        left, right, up, down = cal_bound(img, center, Rx, Ry)  # 获得矩阵的四个点的坐标
        pixNum = (down - up + 1) * (right - left + 1)  # 当前总像素点
        if pixNum == temp_pixNum:
            return Rx, Ry
        # 计算异类点个数
        count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])
        temp_purity = 1 - count / pixNum
        var = np.var(img[up:down + 1, left:right + 1])
        temp_pixNum = pixNum

        if temp_purity > purity and var < var_threshold:
            if purity < 0.99:
                purity = purity * 1.005
            else:
                purity = 0.99
            flag = True
        else:
            flag = False

        if flag == False and item_count % 2 != 0:
            flag_x = False
            Rx -= 1

        if flag == False and item_count % 2 == 0:
            flag_y = False
            Ry -= 1

        if flag_x == False and flag_y == False:
            return Rx, Ry

def downSample(center, h, w):
    newCenters = []
    minCenterLoc = []
    minSeqCenterLoc = []

    disThera=0
    for i in range(len(center)):
        minDis = 1e9
        minLoc = -1
        minSeqDis = 1e9
        minSeqLoc = -1
        for j in range(i + 1, len(center)):
            center_1 = center[i]
            center_2 = center[j]
            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[1] + center_2[1] and (
                    abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[2] + center_2[2]:
                dis = abs(center_1[0][0] - center_2[0][0]) - 1 + abs(center_1[0][1] - center_2[0][1]) - 1
                if (dis < minDis):
                    minDis = dis
                    minLoc = j
            else:
                pass

        minCenterLoc.append(minLoc)
        minSeqCenterLoc.append(minSeqLoc)
    for i in range(len(center)):
        if (i not in minCenterLoc):
            if (minCenterLoc[i] == -1):
                newCenters.append(center[i])
            else:
                center_1 = center[i]
                center_2 = center[minCenterLoc[i]]
                tempCenter = [[0, 0], 0, 0]
                # print(center_1)
                tempCenter[0][0] = int((center_1[0][0] + center_2[0][0]) / 2)
                tempCenter[0][1] = int((center_1[0][1] + center_2[0][1]) / 2)
                tempCenter[1] = int((center_1[1] + center_2[1] + abs(center_1[0][0] - center_2[0][0]) - 1) / 2)
                tempCenter[2] = int((center_1[2] + center_2[2] + abs(center_1[0][1] - center_2[0][1]) - 1) / 2)
                newCenters.append(tuple(tempCenter))
                # print(tuple(tempCenter))
                if (tempCenter[1] < 0 or tempCenter[2] < 0 or tempCenter[1] > 100 or tempCenter[2] > 100):
                    assert f"Rx<0 or Ry < 0 or Rx > width or Ry > height {tempCenter[1]}, {tempCenter[2]}"
    print("newCenters: " + str(len(newCenters)) + "个粒矩")
    return newCenters

def downSeqSample(center, h, w, disThera=4):
    newCenters = []
    minCenterLoc = []
    minSeqCenterLoc = []

    # disThera=4
    for i in range(len(center)):
        minDis = 1e9
        minLoc = -1
        minSeqDis = 1e9
        minSeqLoc = -1
        center_1 = center[i]
        # if (center_1[1]*center_1[2] > 2*2):
        #     continue
        for j in range(i + 1, len(center)):
            center_2 = center[j]
            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[1] + center_2[1] and (
                    abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[2] + center_2[2]:
                pass
            else:
                pass
            dis = abs(center_1[0][0] - center_2[0][0]) - 1 + abs(center_1[0][1] - center_2[0][1]) - 1
            if (dis < minSeqDis and dis < disThera):
            # if (dis < minSeqDis):
                minSeqDis = dis
                minSeqLoc = j

        minCenterLoc.append(minLoc)
        minSeqCenterLoc.append(minSeqLoc)
    for i in range(len(center)):
        if (i not in minSeqCenterLoc):
            if (minSeqCenterLoc[i]==-1):
                newCenters.append(center[i])
            else:
                center_1 = center[i]
                center_2 = center[minSeqCenterLoc[i]]
                tempCenter = [[0, 0], 0, 0]
                # print(center_1)
                tempCenter[0][0] = int((center_1[0][0] + center_2[0][0]) / 2)
                tempCenter[0][1] = int((center_1[0][1] + center_2[0][1]) / 2)
                tempCenter[1] = int((center_1[1] + center_2[1] + abs(center_1[0][0] - center_2[0][0]) - 1) / 2)
                tempCenter[2] = int((center_1[2] + center_2[2] + abs(center_1[0][1] - center_2[0][1]) - 1) / 2)
                newCenters.append(tuple(tempCenter))
                if (tempCenter[1] < 0 or tempCenter[2] < 0 or tempCenter[1] > 100 or tempCenter[2] > 100):
                    assert f"Rx<0 or Ry < 0 or Rx > width or Ry > height {tempCenter[1]}, {tempCenter[2]}"
    print("newCenters: " + str(len(newCenters)) + "个粒矩")
    return newCenters

def img2graph(img, purity=0.7, threshold=10, var_threshold=30):
    """ 本函数用于对图片进行基础粒矩聚类，得到基础粒矩列表以及基础图进行后续操作，例如：可视化，上下采样，旋转，反转等。
    :param img: 输入图片
    :param purity:
    :param threshold:
    :param var_threshold:
    :return: 粒矩列表：center，图：g
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 彩色图转为灰度图
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_label = np.zeros(img.shape)  # 创建label矩阵
    img_grad = get_grad(img)  # 计算梯度图
    max_Grad = img_grad.max()  # 计算梯度图最大值
    center = []  # 创建中心列表
    center_count = 0  # 中心点个数计数

    start = time.time()
    while 0 in img_label: # 存在没有被划分的点
        # 选择一个梯度最小且没有被划分过的点为中心点
        temp_center = center_select(img_grad, img_label)
        # 计算半径 Rx, Ry -> x 方向的半径与 y 方向的半径
        Rx, Ry = cal_Radius(img, temp_center, purity, threshold, var_threshold)
        # 计算实际的矩形在图片中的位置(存在粒矩大小超出图像范围，所以不能直接用半径进行切片)，方便后续使用切片进行特征提取等操作
        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)
        # 添加粒矩 （[x, y], Rx, Ry） 一个粒矩一个元组.最基础特征:中心点坐标， Rx， Ry
        center.append((temp_center, Rx, Ry)) # 粒矩存储方式待优化
        # 将本次迭代生成的粒矩包含的像素点标记 （下次迭代就不会选取这些点作为中心点）
        img_label[up:down + 1, left:right + 1] = 1
        # 将本次迭代生成的粒矩包含的像素点对应位置的梯度设为梯度最大值
        img_grad[up:down + 1, left:right + 1] = max_Grad
        # 粒矩计数
        center_count += 1
    end = time.time()
    print("粒矩聚类时间:%.2f秒" % (end - start))
    print("共生成" + str(center_count) + "个粒矩")


    # 聚类完成，开始构建 Graph (本代码使用 networkx 进行 Graph 的构建，后续如果要考虑构图速度也可以使用其他方法)
    # 图的基本组成 -> 节点集和边集：节点集就是所有粒矩的中心点，边集就是判断两个粒矩是否有重叠的像素点，有就将两个粒矩的中心点相连，即这两个节点之间存在边
    # 初始化 Graph
    g = nx.Graph()
    # 1. 添加节点
    for i in range(len(center)):
        g.add_node(str(i))
    # 2. 生成边 (使用粒矩的位置关系进行边的生成)
    for i in range(len(center)):
        for j in range(i + 1, len(center)):
            center_1 = center[i]
            center_2 = center[j]
            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[2] + center_2[2] and (
                    abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[1] + center_2[1]:  # 相接无边
                #  判断两个中心点之间的距离是否小于它们各自半径之和减1。
                #  因为如果重叠的话，边界上的像素被两个矩阵共享，所以将距离减去1。
                g.add_edge(str(i), str(j))
    # 3. 生成 GNN 需要的数据 -> 存有节点的矩阵，大小为 N * F (N 为节点的个数，F 为节点特征的维度)
    #                        边集，大小为 2 * N (N 为边的个数，每一列为该边连接的两个节点在节点矩阵中的索引)
    #                        边特征矩阵 (可选，部分图神经网络可用)
    a = nx.to_numpy_matrix(g)
    # 邻接矩阵 adj （边集）
    adj = a.A
    adj = sp.coo_matrix(adj)
    adj = np.vstack((adj.row, adj.col))

    center_ = np.zeros((len(center), 4))  # 粒矩基础属性 -> 中心坐标, Rx, Ry （不同的数据可手动提取不同的特征）
    # 生成节点属性和粒矩基础属性数组
    for id in range(len(center)):
        # 粒矩基础属性 center_
        center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2]]

    return center, g

from scipy.stats import skew,kurtosis

def extractFeature(img, left, right, up, down):
    nodeFea = []
    pxs = img[up:down + 1, left:right + 1]

    pixel_values = pxs.flatten()
    sorted_values = np.sort(pixel_values)
    # 计算偏度
    skewness = skew(pixel_values)
    kurtosis_value = kurtosis(pixel_values)
    energy = np.sum(pxs ** 2)
    hist, _ = np.histogram(pxs.flatten(), bins=256, range=[0, 256])
    # 归一化直方图，计算概率分布
    p = hist / np.sum(hist)
    # 计算熵
    entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))  # 忽略0概率项
    nodeFea.append(np.max(pixel_values))
    nodeFea.append(np.min(pixel_values))
    nodeFea.append(np.median(sorted_values))
    nodeFea.append(img[int((down-up)/2), int((right-left)/2)])
    nodeFea.append(np.var(pixel_values))
    nodeFea.append(np.std(pixel_values))
    nodeFea.append(np.mean(pixel_values))
    nodeFea.append(skewness)
    nodeFea.append(kurtosis_value)
    nodeFea.append(energy)
    nodeFea.append(entropy)

    # variances = []
    # means = []
    # maxs = []
    # mins = []
    # for row in pxs:
    #     variances.append(np.var(row))
    #     means.append(np.mean(row))
    #     maxs.append(np.max(row))
    #     mins.append(np.min(row))
    #
    # nodeFea.append(np.max(maxs))
    # nodeFea.append(np.min(mins))
    # nodeFea.append(img[int((down-up)/2), int((right-left)/2)])
    # nodeFea.append(np.mean(variances))
    # nodeFea.append(np.mean(means))
    return nodeFea

def getNodeFea(center, img):
    newCenters = []
    for i in range(len(center)):
        center_i = center[i]
        temp_center = center_i[0]
        Rx = center_i[1]
        Ry = center_i[2]
        # print(f"temp_center:{temp_center}, Rx:{Rx}, Ry:{Ry}")
        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)
        # 计算像素特征
        # print(f"left:{left}, right:{right}, up:{up}, down:{down}")
        nodeFea = extractFeature(img, left, right, up, down)
        # 添加粒矩 （[x, y], Rx, Ry） 一个粒矩一个元组.最基础特征:中心点坐标， Rx， Ry
        newCenters.append((temp_center, Rx, Ry, nodeFea))  # 粒矩存储方式待优化
    return newCenters

def img_resize(image, width_new=224, height_new=224):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    # width_new = 224
    # height_new = 224
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new

def imgToNXGraph(img_name, rH=224, rW=224):
    # img_name = "D:\code\python\granular_ball_img2graph\SLIC.jpg"  # 图片路径
    # img_name = "SLIC.jpg"  # 图片路径
    # img_name = "abc.png"  # 图片路径
    RGB_img = cv2.imread(img_name)  # 读取RGB图用来进行特征提取
    # RGB_img = np.transpose(RGB_img, (2, 0, 1))
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)  # 读取灰度图
    # img = cv2.resize(img, (rH, rW))
    img = img_resize(img, width_new=rW, height_new=rH)
    (height, width) = img.shape
    print(f"height:{height}, width:{width}")
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_label = np.zeros(img.shape)  # 创建label矩阵
    img_grad = get_grad(img)  # 计算梯度图
    max_Grad = img_grad.max()  # 计算梯度图最大值
    center = []  # 创建中心列表
    center_count = 0  # 中心点个数计数

    # 参数设置
    purity = 0.7  # 纯度：粒矩中同类点的占比需要大于设定的纯度
    threshold = 30  # 灰度阈值：同类点与中心的灰度值的差值需要小于设定的灰度阈值
    var_threshold = 50  # 方差阈值：粒矩中所有点的方差需要小于设定的方差阈值

    start = time.time()
    while 0 in img_label:  # 存在没有被划分的点
        # 选择一个梯度最小且没有被划分过的点为中心点
        temp_center = center_select(img_grad, img_label)
        # 计算半径 Rx, Ry -> x 方向的半径与 y 方向的半径
        Rx, Ry = cal_Radius(img, temp_center, purity, threshold, var_threshold)
        # 计算实际的矩形在图片中的位置(存在粒矩大小超出图像范围，所以不能直接用半径进行切片)，方便后续使用切片进行特征提取等操作
        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)
        # 计算像素特征
        # nodeFea = extractFeature(img, left, right, up, down)
        # 添加粒矩 （[x, y], Rx, Ry） 一个粒矩一个元组.最基础特征:中心点坐标， Rx， Ry
        # center.append((temp_center, Rx, Ry, nodeFea))  # 粒矩存储方式待优化
        if(Rx<0 or Ry < 0 or Rx > width or Ry > height):
            assert "Rx<0 or Ry < 0 or Rx > width or Ry > height"
        center.append((temp_center, Rx, Ry))  # 粒矩存储方式待优化
        # 将本次迭代生成的粒矩包含的像素点标记 （下次迭代就不会选取这些点作为中心点）
        img_label[up:down + 1, left:right + 1] = 1
        # 将本次迭代生成的粒矩包含的像素点对应位置的梯度设为梯度最大值
        img_grad[up:down + 1, left:right + 1] = max_Grad
        # 粒矩计数
        center_count += 1
    end = time.time()
    print("粒矩聚类时间:%.2f秒" % (end - start))
    print("共生成" + str(center_count) + "个粒矩")
    center = center[::-1]
    # center = downSample(center, height, width)
    # center = downSample(center, height, width)
    # center = downSeqSample(center, height, width, disThera=3)
    # center = downSeqSample(center, height, width, disThera=5)
    # # center = center[::-1]
    # # center = downSeqSample(center, height, width, disThera=10)
    # # center = downSeqSample(center, height, width, disThera=15)
    # # center = center[::-1]
    # # center = downSeqSample(center, height, width, disThera=20)
    # # center = downSample(center, height, width)
    # # center = downSeqSample(center, height, width, disThera=10)
    while (len(center) > 2048):
        center = downSample(center, height, width)
        center = downSeqSample(center, height, width, disThera=3)
        center = center[::-1]
    # #     center = downSeqSample(center, height, width, disThera=10)
    #     center = downSeqSample(center, height, width, disThera=10)
    #     center = center[::-1]
    #     center = downSample(center, height, width)

    center = getNodeFea(center, img)
    # 聚类完成，开始构建 Graph (本代码使用 networkx 进行 Graph 的构建，后续如果要考虑构图速度也可以使用其他方法)
    # 图的基本组成 -> 节点集和边集：节点集就是所有粒矩的中心点，边集就是判断两个粒矩是否有重叠的像素点，有就将两个粒矩的中心点相连，即这两个节点之间存在边
    # 初始化 Graph
    g = nx.Graph()
    # 1. 添加节点
    for i in range(len(center)):
        c = center[i]
        # g.add_node(str(i), feature=[c[0][0], c[0][1], c[1], c[2]])
        # print(center[3])
        # g.add_node(str(i), feature=[c[0][0], c[0][1], c[1], c[2]] + c[3]+
        #                       [c[0][0]/width, c[0][1]/height, c[1]*c[2]/(width*height)])
        g.add_node(str(i), feature=[c[0][0], c[0][1], c[1], c[2]] + [c[3][4], c[3][6], c[3][7], c[3][0], c[3][1], c[3][-1]])
    # 2. 生成边 (使用粒矩的位置关系进行边的生成)
    edgeNum = 0
    for i in range(len(center)):
        for j in range(i + 1, len(center)):
            center_1 = center[i]
            center_2 = center[j]
            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[2] + center_2[2] and (
                    abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[1] + center_2[1]:  # 相接无边
                # g.add_edge(str(i), str(j))
                if(center_1[1] + center_2[1] != 0):
                    f = (center_1[2] + center_2[2])/(center_1[1] + center_2[1])
                else:
                    f = 0
                iou = abs(center_1[0][0] - center_2[0][0]) * abs(center_1[0][1] - center_2[0][1])
                s1 = 4*center_1[2]*center_1[1]
                s2 = 4*center_2[2]*center_2[1]
                up = min(center_2[0][1],center_1[0][1])
                down = max(center_2[0][1],center_1[0][1])
                left = min(center_2[0][0],center_1[0][0])
                right = max(center_2[0][0],center_1[0][0])
                # print(f"up:{up},down:{down},left:{left},right:{right}")
                if(s1 != 0):
                    r1 = iou/s1
                else:
                    r1=0
                if(s2!= 0):
                    r2 = iou/s2
                else:
                    r2=0

                # g.add_edge(str(i), str(j), feature=[center_1[0][0], center_1[0][1], center_1[1], center_1[2], img[center_1[0][0],center_1[0][1]], s1]+
                #                          [center_2[0][0], center_2[0][1], center_2[1], center_2[2], img[center_2[0][0],center_2[0][1]], s2]+
                #                          [center_1[1] + center_2[1], center_1[2] + center_2[2], iou]+
                #                          extractFeature(img, up, down, left, right)+
                #                          [r1, r2, f])
                g.add_edge(str(i), str(j), feature=[center_1[0][0], center_1[0][1], center_1[1], center_1[2]] +
                                                   [center_2[0][0], center_2[0][1], center_2[1], center_2[2]] +
                                                   [abs(center_1[0][1]-center_2[0][1])+abs(center_1[0][0]-center_2[0][0]), iou])
                edgeNum+=1
    print(f"atoms: {len(center)}, edges: {edgeNum}")
    return center, g

# if __name__ == '__main__':
#
#     # img_name = "D:\code\python\granular_ball_img2graph\SLIC.jpg"  # 图片路径
#     # img_name = "SLIC.jpg"  # 图片路径
#     img_name = "abc.png"  # 图片路径
#     RGB_img = cv2.imread(img_name)  # 读取RGB图用来进行特征提取
#     RGB_img = np.transpose(RGB_img, (2, 0, 1))
#     img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)  # 读取灰度图
#     img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
#     img_label = np.zeros(img.shape)  # 创建label矩阵
#     img_grad = get_grad(img)  # 计算梯度图
#     max_Grad = img_grad.max()  # 计算梯度图最大值
#     center = []  # 创建中心列表
#     center_count = 0  # 中心点个数计数
#
#     # 参数设置
#     purity = 0.7 # 纯度：粒矩中同类点的占比需要大于设定的纯度
#     threshold = 30 # 灰度阈值：同类点与中心的灰度值的差值需要小于设定的灰度阈值
#     var_threshold = 50 # 方差阈值：粒矩中所有点的方差需要小于设定的方差阈值
#
#     start = time.time()
#     while 0 in img_label: # 存在没有被划分的点
#         # 选择一个梯度最小且没有被划分过的点为中心点
#         temp_center = center_select(img_grad, img_label)
#         # 计算半径 Rx, Ry -> x 方向的半径与 y 方向的半径
#         Rx, Ry = cal_Radius(img, temp_center, purity, threshold, var_threshold)
#         # 计算实际的矩形在图片中的位置(存在粒矩大小超出图像范围，所以不能直接用半径进行切片)，方便后续使用切片进行特征提取等操作
#         left, right, up, down = cal_bound(img, temp_center, Rx, Ry)
#         # 添加粒矩 （[x, y], Rx, Ry） 一个粒矩一个元组.最基础特征:中心点坐标， Rx， Ry
#         center.append((temp_center, Rx, Ry)) # 粒矩存储方式待优化
#         # 将本次迭代生成的粒矩包含的像素点标记 （下次迭代就不会选取这些点作为中心点）
#         img_label[up:down + 1, left:right + 1] = 1
#         # 将本次迭代生成的粒矩包含的像素点对应位置的梯度设为梯度最大值
#         img_grad[up:down + 1, left:right + 1] = max_Grad
#         # 粒矩计数
#         center_count += 1
#     end = time.time()
#     print("粒矩聚类时间:%.2f秒" % (end - start))
#     print("共生成" + str(center_count) + "个粒矩")
#
#     # 聚类完成，开始构建 Graph (本代码使用 networkx 进行 Graph 的构建，后续如果要考虑构图速度也可以使用其他方法)
#     # 图的基本组成 -> 节点集和边集：节点集就是所有粒矩的中心点，边集就是判断两个粒矩是否有重叠的像素点，有就将两个粒矩的中心点相连，即这两个节点之间存在边
#     # 初始化 Graph
#     g = nx.Graph()
#     # 1. 添加节点
#     for i in range(len(center)):
#         g.add_node(str(i))
#     # 2. 生成边 (使用粒矩的位置关系进行边的生成)
#     for i in range(len(center)):
#         for j in range(i + 1, len(center)):
#             center_1 = center[i]
#             center_2 = center[j]
#             if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[2] + center_2[2] and (
#                     abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[1] + center_2[1]:  # 相接无边
#                 g.add_edge(str(i), str(j))
#     # 3. 生成 GNN 需要的数据 -> 存有节点的矩阵，大小为 N * F (N 为节点的个数，F 为节点特征的维度)
#     #                        边集，大小为 2 * N (N 为边的个数，每一列为该边连接的两个节点在节点矩阵中的索引)
#     #                        边特征矩阵 (可选，部分图神经网络可用)
#     a = nx.to_numpy_matrix(g)
#     # 邻接矩阵 adj （边集）
#     adj = a.A
#     adj = sp.coo_matrix(adj)
#     adj = np.vstack((adj.row, adj.col))
#
#     center_ = np.zeros((len(center), 4))  # 粒矩基础属性 -> 中心坐标, Rx, Ry （不同的数据可手动提取不同的特征）
#     # 生成节点属性和粒矩基础属性数组
#     for id in range(len(center)):
#         # 粒矩基础属性 center_
#         center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2]]
#
#     # centre_ -> 点集
#     # adj -> 边集
#     # 至此就将一张图片转为了一个图数据