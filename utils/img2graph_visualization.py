import matplotlib.pyplot as plt

from img2graphOrigin import *
random.seed(0)

def RGBToGray(img_name):
    RGB_img = cv2.imread(img_name)
    gray_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2GRAY)
    return gray_img

def img2graphVisualize(img_name):
    # img_name = "frame6895262453882572046_0000003.jpg"  # 图片路径
    # rH, rW = 224, 224
    img = cv2.imread(img_name)
    rH, rW, _ = img.shape
    name = img_name.split(".")[0]
    ext = img_name.split(".")[1]
    RGB_img = cv2.imread(img_name)
    RGB_img = img_resize(RGB_img, rH, rW)

    h, w = RGB_img.shape[0:2]
    center, g = imgToNXGraph(img_name, rH, rW)

    # 粒矩在原图片上的可视化
    for v in center[:150]:
        y, x, Rx, Ry = v[0][0], v[0][1], v[1], v[2]
        cv2.rectangle(RGB_img, (x - Rx, y - Ry), (x + Rx, y + Ry), (0, 255, 0), 1)
    plt.imshow(cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    cv2.imwrite(name+"_" + str(len(center)) + f".{ext}", RGB_img)

    # 图的可视化
    # 1. 确定每个中心的在 graph 中的位置
    pos_dict = {}
    for i in range(len(center)):
        pos_dict[str(i)] = [center[i][0][1], w - center[i][0][0]]
        # print(f"{i}: {pos_dict[str(i)]}")
    print(f"dict len: {len(pos_dict.keys())}")

    fig, ax = plt.subplots()
    nx.draw(g, ax=ax, pos=pos_dict, with_labels=False, width=0.2, edge_color='limegreen', node_color='black', node_size=0.5)  # 设置颜色
    plt.savefig(name+"_graph.svg")
    plt.show(block=True)


# def img_resize(image):
#     height, width = image.shape[0], image.shape[1]
#     # 设置新的图片分辨率框架
#     width_new = 512
#     height_new = 256
#     # 判断图片的长宽比率
#     if width / height >= width_new / height_new:
#         img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
#     else:
#         img_new = cv2.resize(image, (int(width * height_new / height), height_new))
#     return img_new


# img = cv2.imread('lena_test.jpg')
# img_new = img_resize(img)

if __name__ == '__main__':
    img_name = r"D:\Documents\Post-Lab\Papers\AAAI2025-change\code\visualize\data\image0004829___1.jpg"
    # img_name = r"D:\Documents\Post-Lab\Papers\AAAI2025-change\code\visualize\image0000653.jpg"
    img2graphVisualize(img_name)
    # gray_img = RGBToGray(img_name)
    # gray_path = img_name.split(".")[0] + "_gray." + img_name.split(".")[1]
    # cv2.imwrite(gray_path, gray_img)
