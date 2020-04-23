# import os
# #image_path = '/home/yqw/Project/yt-image-classify/Picture/training20180501/dirty'
# image_path='E:/OneDrive - bupt.edu.cn/lab/分割/output/CRA_ori'
# print("a")
# for root, dirs, files in os.walk(image_path):
#     print("root:"+root)
#     print(dirs)
#     print(files)
#


# import os
# image_path = '/home/yqw/Project/yt-image-classify/Picture/training20180501/dirty'
# root = [x[0] for x in os.walk(image_path)]
# dirs = [x[1] for x in os.walk(image_path)]
# images = [x[2] for x in os.walk(image_path)]
# # 遍历所有文件,因为images是一个嵌套list
# def printList(list1):
#     for elements in list1:
#         if isinstance(elements, list) or isinstance(elements, tuple):
#             printList(elements)  # 递归调用函数本身进行深层次的遍历
#         elif isinstance(elements, dict):
#             for i in elements.items():
#                 print(i)
#         else:
#             print(elements)


import os

for root, dirs, files in os.walk("E:/OneDrive - bupt.edu.cn/lab/分割/output/CRA_ori"):
    for name in files:
        print (name)
    for dir in dirs:
        print (dir)
    for rot in root:
        print (rot)