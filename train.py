import os
import torch
import argparse
import random
import numpy as np
from solver.Solver import Model_Solver

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.deterministic = True


def main(config):
    folder_path = {
        'sjtu': 'your dataset path/SJTU-PCQA/',
        'wpc': 'your dataset path/WPC/',
    }
    SRCC_all = np.zeros(config.train_test_num, dtype=np.float64)
    PLCC_all = np.zeros(config.train_test_num, dtype=np.float64)
    KRCC_all = np.zeros(config.train_test_num, dtype=np.float64)
    RMSE_all = np.zeros(config.train_test_num, dtype=np.float64)
    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i + 1))

        s = Model_Solver(config, folder_path[config.dataset])
        SRCC_all[i], PLCC_all[i], KRCC_all[i], RMSE_all[i] = s.train()

    SRCC_med = np.median(SRCC_all)
    PLCC_med = np.median(PLCC_all)
    KRCC_med = np.median(KRCC_all)
    RMSE_med = np.median(RMSE_all)
    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f,\tmedian KRCC %4.4f,\tmedian RMSE %4.4f'
          % (SRCC_med, PLCC_med, KRCC_med, RMSE_med))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='sjtu', help='')
    parser.add_argument('--resume', dest='resume', type=bool, default=False, help='')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='Epochs for training')
    parser.add_argument('--image_size', dest='image_size', type=int, default=224, help='')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')
    parser.add_argument('--model_name', dest='model_name', type=str, default="GC-PCQA", help='')
    parser.add_argument('--split', dest='split', type=int, default=1, help='SJTU:[1-9],WPC:[1-5],SIAT:[1-5]')
    config = parser.parse_args()
    main(config)
