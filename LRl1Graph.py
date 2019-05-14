# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\4\3 0003 10:54:54
# File:         LRl1Graph.py
# Software:     PyCharm
#------------------------------------

import numpy as np
import math
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import minmax_scale

def f(X, B, alpha, L, lamda, gamma):
    '''定义目标函数文中的Eq.(3)'''
    value = 0
    for i in range(X.shape[1]):
        value = value + math.pow(np.linalg.norm(X[:,i]-np.dot(B,alpha[:,i])),2) + lamda*np.linalg.norm(alpha[:,i],1)
    value = value + gamma*np.trace(np.dot(alpha,np.dot(L,alpha.transpose())))
    return value

def sum_alpha(alpha, L):
    #此函数的主要功能是计算solve()中导数的第三项
    value = np.zeros((alpha.shape[0],alpha.shape[1]))
    for i in range(len(L)):
        value = value + L[i]*alpha
    return value

def solve(X, B, alpha, L, lamda, gamma):#求解参数alpha
    f_best = np.inf  # 初始化目标函数值,记录最佳的目标函数值
    alpha_before = alpha  # 记录迭代前一次的alpha矩阵的值
    while f_best >= f(X, B, alpha, L, lamda, gamma):
        '''
         Eq(3)式关于\alpha _{\cdot i}求导后的导数为：
         -2B^{T}x_{i}-2B^{T}B\alpha _{\cdot i}+\lambda\cdot  \frac{\alpha _{\cdot i}}{\left \| \alpha _{\cdot i} \right \|_{1}}+\gamma \cdot \sum _{j=1}^{n}L_{ji}\cdot \alpha _{\cdot i}
        '''
        for i in range(alpha.shape[1]):#本程序中使用的是梯度下降法来求解alpha
            alpha[:,[i]] = alpha[:,[i]] + 2*np.dot(B.transpose(),X[:,[i]])+2*np.dot(np.dot(B.transpose(),B),alpha[:,[i]]) -lamda*alpha[:,[i]]/(np.linalg.norm(alpha[:,[i]],1) + 0.000000001) - gamma*sum_alpha(alpha[:,[i]],L[:,i])
        tempvalue = f(X, B, alpha, L, lamda, gamma)
        if tempvalue < f_best:#目标函数值下降,进行下一次迭代
            f_best = tempvalue
            alpha_before = alpha
        else:
            alpha = alpha_before
            break
    return alpha


def LRl1Graph(data, k):
    '''
    Yang Y, Wang Z, Yang J, et al. Data clustering by laplacian regularized l1-graph[C]//Twenty-Eighth AAAI Conference on Artificial Intelligence. 2014: 3148-3149.
    本算法主要执行的是采用稀疏学习L1-图的数据聚类算法
    data: 输入数据,每一行代表一个样本,每一列代表一个特征
    k: 为谱聚类的数目
    返回的是一个聚类结果,list类型1*k,每个list单元即保存有一个类簇的数据的序号
    '''
    #设置相关的参数,按照文中实验设置
    N_MAX = 2
    lamda = 0.1
    gamma = 30
    X = data.transpose()#将数据转换成列形式
    row = data.shape[0]#数据的样本数
    col = data.shape[1]#数据的维数
    W = np.zeros((row, row))#初始化权值矩阵
    D = np.zeros((row, row))#初始化对角矩阵
    for i in range(row):#计算每对样本之间的相似度
        for j in range(row):
            W[i,j] = math.exp(-np.linalg.norm(data[i,:] - data[j,:],2)/2)
        value0 = sum(W[i,:])/row
        for j in range(row):
            if W[i,j] <= value0:
                W[i, j] = 0
        D[i,i] = sum(W[i,:])#计算对角矩阵的元素
    L = D - W#计算拉普拉斯矩阵
    alpha = np.zeros((col + row, row))#初始化alpha矩阵
    I = np.eye(col)#单位矩阵
    B = np.append(X, I, axis=1)
    for i in range(N_MAX):
        #第一步优化Eq.(3)
        alpha = solve(X, B, alpha, L, lamda, gamma)#求解出优化问题
        W = (alpha[0:row,:]+alpha[0:row,:].transpose())/2#更新权值矩阵W
        for j in range(row):#更新度矩阵
            D[j,j] = sum(W[:,j])
        L = D - W#更新拉普拉斯矩阵
    #根据谱聚类算法来对W进行聚类
    W = minmax_scale(W)
    model = SpectralClustering(n_clusters=k,gamma=1.4,assign_labels='kmeans',affinity= "precomputed")
    result = model.fit_predict(W)
    #result = model.labels_#获取最终的类标签
    result = np.array([result])#最终的结果应该有第0行形式
    return result#返回聚类结果







