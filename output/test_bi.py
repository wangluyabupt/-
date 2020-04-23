#!/usr/bin/env python

import time
from pathlib import Path
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys########################
# sys.path.append("../")
# sys.path.append("./")
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from models_bi import tiramisu
from utils_bi import training as train_utils
from utils_bi.testing import BinaryImgModel



class BinaryImgResult():
    def __init__(self, src_path, dst_path):
        self.binary_img_model = BinaryImgModel()
        # src_path 、 dst_path 均为图片文件夹 + 名称
        self.src_path = src_path
        self.dst_path = dst_path

        normalize = transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
        self.Transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        pass

    def __del__(self):
        pass

    def main(self):

        # 通过numpy 实现对于单张图片进行处理，处理后释放cuda，可以解决上述问题。
        WEIGHTS_PATH = './weights_bi/'

        # test_data = Image.open(self.src_path)
        test_data = cv2.imread(self.src_path)
        # print("test_data.src_shape ::",test_data.shape)

        if isinstance(test_data, np.ndarray):
            test_data = Image.fromarray(test_data)
        data = self.Transforms(test_data)
        data = data.numpy()
        data = np.expand_dims(data, axis=0)  # expand dim;
        # print("test_data.new_shape: {}".format(data.shape))
        data = torch.from_numpy(data)  # convert to Variable;

        # input = Variable(data, volatile=True)

        input = data
        # print('test_data.new_torch_shape::',input.shape)
        # input = Variable(data)
        # input = input.cuda()

        # model = getattr(models_bi, 'FCDenseNet57')().eval()
        # print("opt.load_model_path: {}".format(WEIGHTS_PATH))
        # model.load(WEIGHTS_PATH)
        # self.binary_img_model.main(model, input, self.dst_path)

        torch.cuda.manual_seed(0)
        model = tiramisu.FCDenseNet57(n_classes=256).cuda()  ###########classes
        model.apply(train_utils.weights_init)
        weight_files = os.listdir(WEIGHTS_PATH)
        # print(len(weight_files))
        for k in range(len(weight_files)):
            model.load_state_dict(torch.load(WEIGHTS_PATH + weight_files[k])['state_dict'])
            self.binary_img_model.main(model, input, self.dst_path)
            # self.binary_img_model_direct.main(model, input, self.dst_path, self.last_frame_flag)



        # if opt.use_gpu:
        #     input = input.cuda()
        #     score = self.model(input)
        #     # probability = t.nn.functional.softmax(score)[:, 1].data.tolist()
        #     probability = t.nn.functional.softmax(score, dim=1).data.tolist()  # modified by myself;
        #     print("prob: {}".format(probability))
        #     return probability

if __name__ == '__main__':
    for root, paths, files in sorted(os.walk('CRA_ori')):
        for file in files:
            if file.find('src') > 0:
                SRC_PATH = os.path.join(root, file)
                DST_PATH = os.path.join('output', root)
                if not os.path.exists(DST_PATH):
                    os.makedirs(DST_PATH)
                print(SRC_PATH)
                print(os.path.join(DST_PATH, file))
                binary_img_res = BinaryImgResult(SRC_PATH, os.path.join(DST_PATH, file))
                binary_img_res.main()


    # binary_img_res = BinaryImgResult(SRC_PATH, DST_PATH)
    # binary_img_res.main()
























