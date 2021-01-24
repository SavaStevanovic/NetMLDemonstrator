import numpy as np
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool
import os
import sys
import warnings
from joblib import parallel_backend

class RunningConfusionMatrix():    
    def __init__(self, labels):
        
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        
    def update_matrix(self, y_true, y_pred):
        self.tp += (y_true * y_pred).sum()
        self.tn += ((1 - y_true) * (1 - y_pred)).sum()
        self.fp += ((1 - y_true) * y_pred).sum()
        self.fn += (y_true * (1 - y_pred)).sum()
    
    
    def compute_current_mean_intersection_over_union(self):
        epsilon = 1e-7
    
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = 2* (precision*recall) / (precision + recall + epsilon)

        return f1