import json


class TrainingConfiguration(object):
    def __init__(self, learning_rate=0.0003, iteration_age=0, best_metric=-float('inf'), epoch=1):
        self.learning_rate = learning_rate
        self.iteration_age = iteration_age
        self.best_metric = best_metric
        self.epoch = epoch
        self.steps_done = 0
        self.BATCH_SIZE = 10
        self.GAMMA = 0.99
        self.EPS_START = 1.0
        self.EPS_END = 0.01
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        self.LOWER_LEARNING_PERIOD = 1000
        self.EPOCHS = 50000

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value)
