import os
from os import listdir
from PIL import Image
import shutil
import numpy as np
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
from utils_bi import training as train_utils


class BinaryImgModel:
    def __init__(self):
        pass

    def __del__(self):
        pass

    def recover_pic(self,label):
        img = np.array(Image.new('RGB', (label.shape[0], label.shape[1]), (0, 0, 0)))
        for i in range(0, label.shape[0]):
            for j in range(0, label.shape[1]):
                if (label[i, j] == 0):
                    img[i, j, 0:3] = [0, 0, 0]
                else:
                    img[i, j, 0:3] = [255, 255, 255]

        return img


    # def cal_acc(self,gt_path, pic_path):
    #     pr = 0.0
    #     re = 0.0
    #     files = os.listdir(gt_path)
    #     for k in range(len(files)):
    #         gt = np.array(Image.open(gt_path+ files[k]))
    #         pic = np.array(Image.open(pic_path + files[k]))
    #         pr_count = 0
    #         pr_total = 0
    #         re_count = 0
    #         re_total = 0
    #
    #         for i in range(0,gt.shape[0]):
    #             for j in range(0,gt.shape[1]):
    #                 if(gt[i,j] == 0 and pic[i, j] == 0):
    #                     continue
    #                 if(pic[i, j] != 0):
    #                     pr_total += 1
    #                     if(gt[i, j] == pic[i, j]):
    #                         pr_count += 1
    #                 if(gt[i, j] != 0):
    #                     re_total += 1
    #                     if(gt[i, j] == pic[i, j]):
    #                         re_count += 1
    #
    #         pr += (float(pr_count)/pr_total)
    #         re += (float(re_count)/re_total)
    #         print (files[k])
    #         print ([float(pr_count)/pr_total, float(re_count)/re_total])
    #     pr = pr/len(files)
    #     re = re/len(files)
    #     f1 = 2 * pr * re / (pr + re)
    #     return [pr, re, f1]

    def main(self,model,loader,path_result_save):
        # k = 0
        # print ('001')
        for inputs_ in loader:
            # print ('002')
            # data = Variable(inputs_.cuda(), requires_grad=True)#volatile=True)
            data = inputs_
            # print('data.src_shape::', data.shape)
            data = np.expand_dims(data, axis=0)
            data = torch.from_numpy(data)
            data = Variable(data.cuda(), requires_grad=True)#volatile=True)

            # print('data.new_shape::',data.shape)
            # print ('003')
            output = model(data)
            # print ('004')
            pred = train_utils.get_predictions(output)
            # print ('005')
            pred_ = pred[0].numpy()
            # print ('006')
            out_uint8 = np.array(pred_, dtype=np.uint8)
            # save_pic = Image.fromarray(out_uint8)
            # save_pic.save(RESULTS_PATH + files[k].split('.')[0] + '.png')
            save_pic_1 = self.recover_pic(out_uint8)
            save_pic_1_ = Image.fromarray(save_pic_1)

            save_pic_1_.save(path_result_save)
            #save_pic_1_.save( RESULT_PATH + files[k].split('.')[0] + '.png')

            print ('Seg Binary Img Done !!!')
            # print ('processing %d'%k)
            # k+=1












