#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:20:46 2018

@author: root
"""

import numpy as np
import pandas as pd
A_anwser = pd.read_csv('../data/f_answer_a_20180306.csv',header=None)
A_test = pd.read_csv('../data/f_test_a_20180204.csv',encoding='gb2312')
orig_train = pd.read_csv('../data/f_train_20180204.csv',encoding='gb2312')
A_test['label'] = A_anwser[0]
new_train = pd.concat([orig_train,A_test],ignore_index=True)
new_train.to_csv("../data/d_train_20180307.csv",index=False,encoding='gb2312')