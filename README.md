# GC-PCQA
This is the code for "No-reference Point Cloud Quality Assessment via Graph Convolutional Network", and it has been accepted by IEEE Transactions on Multimedia.

# How to start with the code?

## Environment
python 3.7  
pytorch 1.13.1  
pytorch-cuda 11.6  
scipy
numpy
pandas  
pillow

## Dataset
The code is tested on the SJTU-PCQA and WPC datasets, which can be downloaded from the following links.  
SJTU-PCQA：[https://smt.sjtu.edu.cn/].  
WPC：[https://github.com/qdushl/Waterloo-Point-Cloud-Database].  
For these datasets, you can use 'utils/get_projection2D.py' to generate multi-view projection images.   
The adjacency matrix based on the angular distance of projections is stored in 'utils/spatial_position.mat'.

## Training and Testing
You can simply train model with the following command:
```
python train.py
```
In particular, you need to set the dataset path and specify the test fold after dataset splitting (for K-fold cross-validation).  
After training, you can perform testing using the saved weight file with the following command:  
```
python test.py
```

## Citation
If you find our work useful, please give us star and cite our paper as:
```
article{chen2024dhcn,
  title={No-Reference Point Cloud Quality Assessment via Graph Convolutional Network}, 
  author={Chen, Wu and Jiang, Qiuping and Zhou, Wei and Shao, Feng and Zhai, Guangtao and Lin, Weisi},
  journal={IEEE Transactions on Multimedia}, 
  year={2024},
}
```
