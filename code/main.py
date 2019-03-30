#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 10:19:56 2018

@author: root
"""

import os
os.system("python addA.py")
print ("addA complete")
os.system("python lgb.py")
print ("lgb complete")
os.system("python XGB_10.py")
print ("XGB_10 complete")
os.system("python xgb_gbdt_f1.py")
print ("xgb_gbdt_f1 complete")
os.system("python xgb_dart_f1.py")
print ("xgb_dart_f1 complete")
os.system("python bagging_model.py")
print ("final complete")
