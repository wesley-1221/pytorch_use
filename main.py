# coding:utf8

import numpy as np
import os
import cv2
from label_process import process_csv
from img_process import VisualizeImage, ProcessImage
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import pandas as pd


LABLE_FILE_PATH = 'label_num_to_disease_map.json'
TRAIN_FILE_PATH = 'train.csv'
SAVE_PATH = 'concat_csv'
IMAGE_PATH = 'train_img'


# cv2 展示图片
def cv_show(popUp_name,img):
    """
    :param popUp_name: 弹窗名
    :param img: 图片
    :return:
    """
    cv2.imshow(popUp_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图片缩放
def resize_image(self, image_size):     # (205, 136) 可以指定大小
    resized_img = cv2.resize(self.image_matrix, image_size)
    return resized_img                  # resized_img改变， 原来image_matrix没有改变

# 入图片id， 查看并展示图片相关信息
def load_img(image_id):
    image_matrix = cv2.imread(os.path.join('process_img', image_id))

    return image_matrix

# 图像分类观察，观察这六种叶片
def show_six_type(original_data, leave_type):
    # 展示某一类树叶样本
    dic_type = [key for key in leave_type.keys()]
    plt.figure(figsize=(12, 6))
    # image_ids = original_data.loc[original_data.label == dic_type[0], 'image_id'][:4]
    image_ids = np.random.choice(original_data.loc[original_data.label == dic_type[0], 'image_id'],4)
    loc = 1
    for image_id in image_ids:
        image_data = plt.imread(os.path.join(IMAGE_PATH, image_id))
        plt.subplot(2, 2, loc)
        loc += 1
        plt.imshow(image_data)
        plt.title(leave_type[dic_type[0]])
    plt.show()


def save_img(img_id, img):
    # if os.path.exists("./process_img"):
    #     print("存在该目录")
    # else:
    #     os.mkdir('./process_img')
    cv2.imwrite(os.path.join('./new/', img_id), img)
    # print("保存成功")


# 判断什么时候数据增强，什么时候采用滤波
# 首先查看样本分布情况 可以看出样本不均衡
def sample_distribution(original_data, label_names):
    sns.countplot(original_data.label.map(lambda x: label_names[str(x)]))
    plt.xticks(rotation=20)
    plt.show()

# 数据增强
def add_sample(original_data, process_img):        # 处理数据对象
    dic = {}
    sample = original_data.label.value_counts().reset_index(name="total")
    distance = sample.total.max() - sample.total
    sample['distance'] = distance
    new_data = sample.loc[:, ['index','distance']]
    print(new_data)
    # 用来确定哪一列需要做数据增强
    for i in new_data.index:
        distance_num = new_data.loc[i, ['index', 'distance']].values[1]
        if distance_num != 0:      # 对这些类型进行图像增强，增加distance这么多
            type = new_data.loc[i, ['index', 'distance']].values[0]   # 类别
            # 获取到各类型的id
            img_id = original_data.loc[original_data.label == type, 'image_id']    # 筛选出符合条件的img_id进行缩放
            # 下面对图片进行添加样本操作
            img_list = np.random.choice(img_id, distance_num // 4)
            for j in range(len(img_list)):     # 一张图片可加7个样本
                image = load_img(img_list[j])
                img_dic = process_img.add_sample_rotate(image)
                save_img("{}{}{}".format(90, j, img_list[j]), img_dic[90])
                save_img("{}{}{}".format(180, j, img_list[j]), img_dic[180])
                save_img("{}{}{}".format(270, j, img_list[j]), img_dic[270])

                img_matrix = process_img.add_sample_affine(image)
                save_img("{}{}".format(j, img_list[j]), img_matrix)

                dic["{}{}{}".format(90, j, img_list[j])] = type
                dic["{}{}{}".format(180, j, img_list[j])] = type
                dic["{}{}{}".format(270, j, img_list[j])] = type
                dic["{}{}".format(j, img_list[j])] = type

    return dic






if __name__ == '__main__':

    # concat_csv, row_csv, label_names = process_csv(TRAIN_FILE_PATH, LABLE_FILE_PATH, SAVE_PATH)
    # print(np.random.choice(row_csv['label']))

    image_matrix = load_img("Train_10.jpg")
    visual_img = VisualizeImage(image_matrix)
#
    print(visual_img.structure_rgb_df())
    print("means:", visual_img.avg_rgb())
    visual_img.pixel_distribution()
#     # visual_img.rgb_boxplot()
#     # visual_img.rgb_show()
#
#     # process_img = ProcessImage(image_matrix)
#     # resized_img = process_img.resize_image((224, 224))
#
#     # 增强
#     # rotate_dic = process_img.add_sample_rotate(resized_img)
#     # affine_img = process_img.add_sample_affine(resized_img)
#     # process_img.add_sample_image(resized_img)
#
#     # 过滤
#     # process_img.median_filter(resized_img)     # 中值
#     # process_img.blur_filter(resized_img)       # 双边
#     # process_img.remove_color_nosie(resized_img)
#
#     # # 随机展示一种类别图片
#     # leave_type = np.random.choice(
#     #     [
#     #         {0: "Cassava Bacterial Blight (CBB)"},
#     #         {1: "Cassava Brown Streak Disease (CBSD)"},
#     #         {2: "Cassava Green Mottle (CGM)"},
#     #         {3: "Cassava Mosaic Disease (CMD)"},
#     #         {4: "Healthy"},
#     #         {5: "apple leaves(AL)"}
#     #     ]
#     #
#     # )
#     # show_six_type(row_csv, leave_type)
#
#
#     # 保存图片
#
#     # save_img("02.jpg", affine_img)
#     # sample_distribution(row_csv, label_names)
#     # add_sample(row_csv, label_names)
# """
#
#
#     # 1. 将所有图片进行一个缩放（224，224）再保存
#     # image_path = "./train_img/"
#     # train = pd.read_csv('./train.csv')
#     # # classes = train['image_id'].values
#     # for i in range(len(train)):
#     #     image_id = train['image_id'].values[i]  # 一列
#     #     image_matrix = cv2.imread(os.path.join(image_path, image_id))
#     #     process_img = ProcessImage(image_matrix)
#     #     resized_img = process_img.resize_image((224, 224))
#     #     save_img(image_id, resized_img)
#
#     # # 2. 保存到指定文件夹后对图片进行增强
#     # image_path = "./process_img/"
#     # train = pd.read_csv('./train.csv')
#     # image_matrix = load_img("Train_10.jpg")
#     # process_img = ProcessImage(image_matrix)
#     # label_dic = add_sample(train, process_img)
#     # with open("matrix_dic.txt", "w") as output:
#     #     output.write(str(label_dic))








