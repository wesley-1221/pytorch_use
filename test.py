# coding:utf8

import pandas as pd


# with open('matrix_dic.txt', 'r', encoding='utf-8') as fp:
#     x = fp.read()
# label = []
# img = []
# y = eval(x)
# for i in y.values():
#     img.append(i)
#
# for j in y.keys():
#     label.append(j)
#
# d = {
#     'image_id': label,
#     'label': img
#
#      }
# frame = pd.DataFrame(d)
# frame.to_csv("addSample.csv", index=False)



x = pd.read_csv("addSample.csv")
y = pd.read_csv("train.csv")
z = pd.concat([x,y], axis=0)
z.to_csv('concat_train.csv', index=False)

x = pd.read_csv("concat_train.csv")

print(x.label.value_counts())

