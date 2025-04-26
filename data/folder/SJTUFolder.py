import torch.utils.data as data
import os
import scipy.io
import scipy.io as scio
import torch
import numpy as np
from numpy import *
from PIL import Image, ImagePath


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class SJTUFolder(data.Dataset):

    def __init__(self, root, index, transform, istrain, config):
        self.istrain = istrain
        self.config = config
        order = ['redandblack', 'Romanoillamp', 'loot', 'soldier', 'ULB Unicorn', 'longdress', 'statue', 'shiva', 'hhi']

        # Split training and test images based on indices
        if istrain:
            index_order = order[:index - 1] + order[index:]
        else:
            index_order = order[index - 1:index]

        # 提取标签
        mos = scipy.io.loadmat(os.path.join(root, 'Final_MOS.mat'))
        labels = mos['Final_MOS'].astype(np.float32)

        self.data = []
        self.dis_path = root + "distortion2D"

        use_number = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
        use_number2 = [31, 34, 37, 40, 43, 46, 49, 52, 55, 58]
        a_path = os.path.join(root + "spatial_position.mat")
        A = scio.loadmat(a_path)["A"]

        for ind in index_order:
            for img_num in range(42):
                file_list = []
                for j in use_number:
                    file_name = ind + "_" + str(img_num) + "_" + str(j) + ".png"
                    file_path = os.path.join(self.dis_path, ind, file_name)
                    file_list.append(file_path)

                for j in use_number2:
                    file_name = ind + "_" + str(img_num) + "_" + str(j) + ".png"
                    file_path = os.path.join(self.dis_path, ind, file_name)
                    file_list.append(file_path)

                order_ind = order.index(ind)
                label = labels[img_num][order_ind]

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
