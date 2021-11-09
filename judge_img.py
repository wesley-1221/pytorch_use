# coding:utf8
import pandas as pd
import cv2
import os
from img_process import ProcessImage
import main
# 用于判断什么时候该增强数据，什么时候该过滤数据
image_matrix = main.load_img("Train_10.jpg")
pro_img = ProcessImage("image_matrix")


image_path = "./process_img/"
train = pd.read_csv('./train.csv')
# classes = train['image_id'].values
for i in range(len(train)):
    image_id = train['image_id'].values[i]  # 一列
    image_data = cv2.imread(os.path.join(image_path, image_id))
    # 异常值处理
    d={'R':image_data[:,:,0].ravel(),
       'G':image_data[:,:,1].ravel(),
       'B':image_data[:,:,2].ravel()}
    frame = pd.DataFrame(d)
    RGB = ['R', 'G', 'B']
    # [frame.describe()[i] for i in RGB]
    for color in RGB:
        interquartile_distance = frame.describe()[color].loc['75%'] - frame.describe()[color].loc['25%']  # 四分位距
        upper_limit = frame.describe()[color].loc['75%'] + 1.5 * interquartile_distance  # 上须
        lower_limit = frame.describe()[color].loc['25%'] - 1.5 * interquartile_distance  # 下须
        # 判断异常值，如果一个通道中有异常值，将整个通道取出
        up_series = frame[color] > upper_limit
        domw_series = frame[color] < lower_limit
        # print((up_series == False).mean(), (domw_series == False).mean())
        if (up_series == False).mean() != 1.0 or (domw_series == False).mean() != 1.0:
            # print("这里对图像进行过滤操作")
            median_martix = pro_img.median_filter(image_data)
            matix = pro_img.remove_color_nosie(median_martix)
            main.save_img(image_id, matix)
        else:
            # print("数据正常")
            main.save_img(image_id, image_data)


