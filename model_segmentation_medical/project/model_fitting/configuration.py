import json


class TrainingConfiguration(object):
    def __init__(
        self,
        learning_rate=0.0001,
        iteration_age=0,
        best_metric=None,
        epoch=0,
        stop_age=10,
    ):
        self.learning_rate = learning_rate
        self.iteration_age = iteration_age
        self.best_metric = best_metric
        self.epoch = epoch
        self.stop_age = stop_age

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value)
