
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import  pickle
import time
import pandas as pd

def visual(X):
    tsne = manifold.TSNE(n_components=2,init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    #'''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    return  X_norm


import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd
import torch

# 从 .pt 文件中加载数据
xPath = r"D:\Documents\Post-Lab\Papers\AAAI2025-change\code\tSNE\SAMM.pt"
# xPath = "hiddenFeatures/RAFDBFeatures/basic/testHiddenFea_epoch1.pt"
yPath = r"D:\Documents\Post-Lab\Papers\AAAI2025-change\code\tSNE\SAMMY.pt"
# accStr = "90.17%"
# accStr = "86.49%"
accStr = "95.96%"
S_X1 = torch.load(xPath)
y_s = torch.load(yPath)
S_X1 = S_X1.cpu().numpy()
y_s = y_s.cpu().numpy()
S_X1 = np.stack(S_X1)
y_s = np.array(y_s)
# 打印形状以验证
print(S_X1.shape)
print(y_s.shape)
class_num = len(set(list(y_s)))
maxX = -100.0
minY = 100.0
# import sys
# sys.exit(0)
###################

maker = 'o'  # 设置散点形状
colors = ['black', 'tomato', 'yellow', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']  # 设置散点颜色

Label_Com = ['S-1', 'T-1', 'S-2', 'T-2', 'S-3',
             'T-3', 'S-4', 'T-4', 'S-5', 'T-5', 'S-6', 'T-6', 'S-7', 'T-7', 'S-8', 'T-8', 'S-9', 'T-9',
             'S-10', 'T-10', 'S-11', 'T-11', 'S-12', 'T-12']  ##图例名称

### 设置字体格式
font1 = {'family': 'Times New Roman',

         'weight': 'bold',
         'size': 32,
         }


def visual(X):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    # '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    return X_norm


def plot_with_labels(S_lowDWeights, Trure_labels, name, colors, maxX, minY):
    plt.cla()  # 清除当前图形中的当前活动轴,所以可以重复利用

    # 降到二维了，分别给x和y
    True_labels = Trure_labels.reshape((-1, 1))

    S_data = np.hstack((S_lowDWeights, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    for index in range(class_num):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker,
                    edgecolors=colors[index], facecolors=colors[index], alpha=0.65)
        maxX = max(max(X), maxX)
        minY = min(min(Y), minY)

    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    #
    plt.title(name, fontsize=32, fontweight='normal', pad=20)

    return maxX, minY

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
maxX, minY = plot_with_labels(visual(S_X1), y_s, '', colors, maxX, minY)

plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None,
                    wspace=0.1, hspace=0.15)
# plt.legend(scatterpoints=1, labels=Label_Com, loc='best', labelspacing=0.4, columnspacing=0.4, markerscale=2,
#            bbox_to_anchor=(0.9, 0), ncol=12, prop=font1, handletextpad=0.1)
plt.text(maxX-0.21, 0.97, accStr, fontdict={'fontsize': 32, 'fontweight': 'bold', 'family': 'serif'})
# plt.savefig('./'+str(sour)+str(tar)+'.png', format='png',dpi=300, bbox_inches='tight')
plt.show()
