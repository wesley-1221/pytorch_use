# coding:utf8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import os

# 需要注意，plt读取是RGB   cv2读取是BGR



# 可视化，展示图片信息   输入一张图片矩阵，查看信息
class VisualizeImage():

    def __init__(self, image_matrix):
        self.image_matrix = image_matrix

    # 返回一个rgb字典
    def structure_rgb_df(self):
        rgb_dict = {'R': self.image_matrix[:, :, 2].ravel(),
                    'G': self.image_matrix[:, :, 1].ravel(),
                    'B': self.image_matrix[:, :, 0].ravel()
                    }
        rgb_df = pd.DataFrame(rgb_dict)
        rgb_columns = rgb_df.columns
        return rgb_df, rgb_columns

    # 计算每一幅图像的 三通道平均值 返回一个列表套字典
    def avg_rgb(self):
        rgb_avg_list = []
        rgb_df, rgb_columns = self.structure_rgb_df()       # 获得rgb dataframe
        for color in rgb_columns:
            rgb_avg_list.append({color: rgb_df[color].mean()})
        return rgb_avg_list

    # 图像像素整体分布（sns.distplot）
    def pixel_distribution(self):
        rgb_df, rgb_columns = self.structure_rgb_df()
        # 依次展示rgb三通道的分布情况
        for color in rgb_columns:
            sns.distplot(rgb_df[color])
            plt.show()

    # 三通道颜色分布对比箱型图
    def rgb_boxplot(self):
        rgb_df, rgb_columns = self.structure_rgb_df()
        # 三个通道的图可以直接绘制出来， 但是会有异常值， 后续处理
        sns.boxplot(data=rgb_df, palette="Set3").set(xlabel='RGB', ylabel='Pixel point')
        plt.show()

    # 用滤镜观察RGB图像, 灰度图像
    def rgb_show(self):
        rgb_df, rgb_columns = self.structure_rgb_df()
        # print(rgb_df['R'].values)
        # print(rgb_df['R'].shape)
        plt.imshow(self.image_matrix[:, :, 2], cmap=plt.cm.Reds)
        plt.show()
        plt.imshow(self.image_matrix[:, :, 1], cmap=plt.cm.Greens)
        plt.show()
        plt.imshow(self.image_matrix[:, :, 0], cmap=plt.cm.Blues)
        plt.show()
        # 灰度处理
        plt.imshow(self.image_matrix.mean(axis=2) / 255., cmap=plt.cm.gray)
        plt.show()



# 所有图片都需要执行的操作， 图像预处理
class ProcessImage():

    def __init__(self, image_matrix):
        self.image_matrix = image_matrix

    # 图片缩放
    def resize_image(self, image_size):     # (205, 136) 可以指定大小
        resized_img = cv2.resize(self.image_matrix, image_size)
        return resized_img                  # resized_img改变， 原来image_matrix没有改变

    # 图像增强（通多观察，样本差异较大，所以部分类别需要做图像增强）  最终需要保存
    # 旋转 执行一次得到三张
    def add_sample_rotate(self, variable_img):
        rotate_dic = {}
        rotate_list = [90, 180, 270]
        rows, cols = variable_img.shape[:2]       # 彩色
        for rotate in rotate_list:
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate, 1)
            rotation_angle = cv2.warpAffine(variable_img, M, (cols, rows))
            rotate_dic[rotate] = rotation_angle
            # cv2.imshow("旋转{}度".format(rotate), rotation_angle)
            # cv2.waitKey(0)
        return rotate_dic

    # 仿射  得到一张
    def add_sample_affine(self, variable_img):
        # 对图像进行变换（三点得到一个变换矩阵）
        # 我们知道三点确定一个平面，我们也可以通过确定三个点的关系来得到转换矩阵
        # 然后再通过warpAffine来进行变换
        rows, cols = variable_img.shape[:2]
        point1 = np.float32([[50, 50], [300, 50], [50, 200]])
        point2 = np.float32([[10, 100], [300, 50], [100, 250]])
        M = cv2.getAffineTransform(point1, point2)
        affine = cv2.warpAffine(variable_img, M, (cols, rows), borderValue=(255, 255, 255))
        # cv2.imshow("仿射", affine)
        # cv2.waitKey(0)
        return affine

    # # 镜像   三张
    # def add_sample_image(self, variable_img):
    #     # 水平镜像
    #     h_flip = cv2.flip(variable_img, 1)
    #     cv2.imshow("Flipped Horizontally", h_flip)
    #     # 垂直镜像
    #     v_flip = cv2.flip(variable_img, 0)
    #     cv2.imshow("Flipped Vertically", v_flip)
    #     # 水平垂直镜像
    #     hv_flip = cv2.flip(variable_img, -1)
    #     cv2.imshow("Flipped Horizontally & Vertically", hv_flip)
    #     cv2.waitKey(0)
    #     return h_flip, v_flip, hv_flip

    # 过滤
    # 中值滤波
    def median_filter(self, variable_img):
        # 5*5 卷积
        median = cv2.medianBlur(variable_img, 5)  # 中值滤波
        # cv2.imshow('median', median)
        # cv2.waitKey(0)
        return median

    # 双边滤波
    def blur_filter(self, variable_img):
        blur = cv2.bilateralFilter(variable_img, 9, 75, 75)
        # cv2.imshow('blur', blur)
        # cv2.waitKey(0)
        return blur

    # cv.fastNlMeansDenoisingColored() 如上所述，它用于消除彩色图像中的噪点。（噪声可能是高斯的）
    def remove_color_nosie(self, variable_img):
        dst = cv2.fastNlMeansDenoisingColored(variable_img, None, 10, 10, 7, 21)
        # plt.subplot(121), plt.imshow(variable_img)
        # plt.subplot(122), plt.imshow(dst)
        # plt.show()
        return dst















