# coding:utf8
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import json


# 'train.csv' 'label_num_to_disease_map.json'  合并
def process_csv(train_file, label_file, save_path):
    """
    :param train_csv: 图片id-标签文件路径
    :param label_json: 标签-图片类型文件路径
    :return:
    """
    row_data = pd.read_csv(train_file)
    # 对csv文件数据进行一个简单查看
    # print(row_data.shape)
    # print(row_data.label.value_counts())

    # csv文件与json文件进行合并
    with open(label_file, 'r', encoding='utf-8') as fp:
        label_names = json.load(fp)

    # print(label_names)
    encoder = OneHotEncoder()                            # 实例化
    matrix = encoder.fit_transform(row_data[['label']])  # 构建标签矩阵
    matrix.toarray()          # 转成array数组
    label_matrix = pd.DataFrame(data=matrix.toarray(), columns=label_names.values())   # 构建DataFrame
    label_data = pd.concat((row_data, label_matrix), axis=1).copy()    # 合并数据
    label_data.drop(labels=['label'], axis=1, inplace=True)            # 剔除列
    label_data.to_csv(save_path, index=False)                    # 保存

    return label_data, row_data, label_names



print(pd.read_csv("concat_csv").head())





