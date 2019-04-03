# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\4\3 0003 15:11:11
# File:         demo.py
# Software:     PyCharm
#------------------------------------

import numpy as np
from scipy.io import loadmat
import math
import LRl1Graph
import Evaluation
from sklearn.preprocessing import minmax_scale
import time

if __name__ == '__main__':
    start = time.time()
    #导入数据
    data = loadmat('Breastw.mat')
    data = data['Breastw']
    data = np.array(data)
    k = 3#数据聚类的类簇数
    col = data.shape[1]#数据的维数
    n = data.shape[0]#数据的样本数目
    label = np.array([data[:,col-1]])-1#获取数据的类标签
    data = data[:,0:col-1]#获取数据部分
    data = minmax_scale(data)#数据归一化处理
    result = LRl1Graph.LRl1Graph(data, k)
    FM, ARI, Phi, Hubert, K, RT, precision, recall, F1, NMI, Q = Evaluation.evaluation(np.dot(data,data.transpose()),result,label)
    print('FM=', FM)
    print('ARI=', ARI)
    print('Hubert=', Hubert)
    print('K=', K)
    print('RT=', RT)
    print('precision=', precision)
    print('recall=', recall)
    print('F1=', F1)
    print('NMI=', NMI)
    print('Q=', Q)
    end = time.time()
    print('算法运行的时间为:',end - start)

