import cv2
import os
import numpy as np

# path1=os.path.abspath('.')   # 表示当前所处的文件夹的绝对路径
# print(path1)
# ##/home/wly/Documents/output



def img_filter(coor, img):
    color = [0, 0, 0]
    padding = 8
    new_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=color)
    block = new_img[coor[0]:coor[0] + 2 * padding, coor[1]:coor[1] + 2 * padding, :]
    dict_color = dict()
    for h in range(block.shape[0]):
        for w in range(block.shape[1]):
            array_tuple = tuple(block[h, w])
            if array_tuple not in dict_color.keys():
                dict_color[array_tuple] = 1
            else:
                dict_color[array_tuple] += 1
    background_tuple = tuple(np.array([0, 0, 0]))
    if background_tuple in dict_color.keys() and len(dict_color.keys()) == 1:
        return background_tuple[::-1]
    else:
        if background_tuple in dict_color.keys():
            dict_color.pop(background_tuple)
        return max(dict_color, key=dict_color.get)[::-1]


def process(path_img_bi, path_img_labeled_raw, path_output, file):
    colors = {
        (0, 0, 0): (0, 0, 0),

        (0, 159, 48): (255, 0, 0),

        (177, 135, 60): (0, 255, 0),
        (255, 111, 72): (255, 0, 255),
        (0, 87, 84): (255, 255, 0),

        (177, 63, 96): (0, 0, 255),
        (255, 39, 108): (0, 128, 255),
        (0, 15, 120): (0, 255, 255),
        (177, 246, 132): (128, 0, 255),
    }

    img_bi = cv2.imread(path_img_bi)
    img_labeled_raw = cv2.imread(path_img_labeled_raw)
    img_labeled_raw = cv2.resize(img_labeled_raw, (512, 512))
    img_filtered = np.zeros((512, 512, 3))

    for h in range(img_bi.shape[0]):
        for w in range(img_bi.shape[1]):
            if not np.array_equal(img_bi[h, w], [0, 0, 0]):
                # if img_bi[h, w][0] == img_bi[h, w][1] == img_bi[h, w][2]:
                #     img_bi[h, w] = [255, 255, 255]
                color_pixel_filtered = img_filter((h, w), img_labeled_raw)
                try:
                    color_pixel_filtered = colors[color_pixel_filtered]
                    color_pixel_filtered = np.array(color_pixel_filtered[::-1])
                    img_filtered[h, w] = color_pixel_filtered
                except KeyError:
                    pass
                    # print('There is no such %s in color dict.' % str(color_pixel_filtered))

    cv2.imwrite(os.path.join(path_output, file), img_filtered)


num_files = 0
for root, path, files in sorted(os.walk('CRA_ori_output')):
    if files:
        num_files += len(files)

for root, path, files in sorted(os.walk('CRA_ori_output')):
    if files:
        for file in files:
            path_img_bi = os.path.join(root, file)
            print(path_img_bi)

            #path_img_labeled_raw = os.path.join('CRA_ori', root.split('\\', 1)[1], file.split('_src_')[0] + '_label_' + file.split('_src_')[1])
            path_img_labeled_raw = os.path.join('CRA_ori' ,root.split('/', 1)[1] , file.split('_src_')[0] +'_label_' +file.split('_src_')[1])
##############################################################
            print(path_img_labeled_raw)

            #path_output = os.path.join('filtered', root.split('\\', 1)[1])
            path_output = os.path.join('filtered', root.split('/', 1)[1])
            print(path_output )
            if not os.path.exists(path_output):
                os.makedirs(path_output)
            process(path_img_bi, path_img_labeled_raw, path_output, file)
            if num_files > 0:
                num_files -= 1
                print('There are %d left.' % num_files)






