import numpy as np
from multiprocessing import Pool


class RunningConfusionMatrix:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update_matrix(self, y_true, y_pred):
        self.tp += (y_true * y_pred).sum()
        self.tn += ((1 - y_true) * (1 - y_pred)).sum()
        self.fp += ((1 - y_true) * y_pred).sum()
        self.fn += (y_true * (1 - y_pred)).sum()

    def compute_metrics(self):
        self.tp = self.tp.item()
        self.tn = self.tn.item()
        self.fp = self.fp.item()
        self.fn = self.fn.item()

        epsilon = 1e-7
        total = self.tp + self.tn + self.fp + self.fn + epsilon

        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        acc = (self.tp + self.tn) / total
        confusion_matrix = np.matrix([[self.tp, self.fn], [self.fp, self.tn]]) / total

        print(confusion_matrix)
        return f1, acc, confusion_matrix
