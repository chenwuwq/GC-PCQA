import torch.utils.data as data
import os
import scipy.io
import scipy.io as scio
import torch
import numpy as np
from PIL import Image, ImagePath
import pandas as pd


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class WPCFolder(data.Dataset):

    def __init__(self, root, index, transform, istrain, config):
        self.istrain = istrain
        self.config = config

        # Split training and test images based on indices
        split_index = ['banana', 'cauliflower', 'mushroom', 'pineapple', 'bag', 'biscuits', 'cake',
                       'flowerpot', 'glasses_case', 'honeydew_melon', 'house', 'pumpkin', 'litchi',
                       'pen_container', 'ping-pong_bat', 'puer_tea', 'ship', 'statue', 'stone', 'tool_box']
        if istrain:
            index_order = split_index[:index * 4 - 4] + split_index[index * 4:]
        else:
            index_order = split_index[index * 4 - 4:index * 4]

        # Extracting labels
        mos = pd.read_excel(os.path.join(root, 'WPC_MOS.xlsx'))
        self.data = []
        self.dis_path = root + "distorted2D"

        use_number = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
        use_number2 = [31, 34, 37, 40, 43, 46, 49, 52, 55, 58]
        a_path = os.path.join(root + "spatial_position.mat")
        A = scio.loadmat(a_path)["A"]

        for i in range(len(mos)):
            ind = None
            file_list = []
            file_name = mos.iloc[i, 1].split(".ply")[0]
            for m in index_order:
                if m in file_name:
                    ind = m
                    break
            if ind is None:
                continue

            for j in use_number:
                if "_pqs_1_qs" in file_name:
                    file_name2 = file_name + "_rounded_" + str(j) + ".png"
                else:
                    file_name2 = file_name + "_" + str(j) + ".png"
                file_path = os.path.join(self.dis_path, ind, file_name2)
                file_list.append(file_path)

            for j in use_number2:
                if "_pqs_1_qs" in file_name:
                    file_name2 = file_name + "_rounded_" + str(j) + ".png"
                else:
                    file_name2 = file_name + "_" + str(j) + ".png"
                file_path = os.path.join(self.dis_path, ind, file_name2)
                file_list.append(file_path)

            label = mos.iloc[i, 2] / 10

            self.data.append((file_list, label, A))

        self.transform = transform
        print("load dataset num:", len(self.data))

    def __getitem__(self, index):

        file_list, label, A = self.data[index]
        imgs = None
        for i in file_list:
            if imgs is None:
                imgs = self.transform(pil_loader(i)).unsqueeze(0)
            else:
                file = self.transform(pil_loader(i)).unsqueeze(0)
                imgs = torch.cat((imgs, file), dim=0)

        A = torch.as_tensor(A)
        return imgs, label, A

    def __len__(self):
        length = len(self.data)
        return length
