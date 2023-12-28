#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/11/8 14:15

@author: AS ME
"""

from args import *
import scipy.io as scio
import numpy as np
arg=arg_parse()
set=Set(arg)
dic=scio.loadmat(set.input_path_data + 'Pw' + '.mat')

data = list(scio.loadmat(set.input_path_data + 'Pw' + '.mat').values())[3]
print(data.shape)
data = np.reshape(data, (-1, 10,set.length))# 【B，L】

data =data[:,0,:]
data =np.reshape(data,[-1])
dic['Pwaa']=data
print(dic['Pwaa'].shape)
scio.savemat(set.input_path_data + 'Pw1' + '.mat',dic)
data = list(scio.loadmat(set.input_path_data + 'Pw1' + '.mat').values())[3]
print(data.shape)