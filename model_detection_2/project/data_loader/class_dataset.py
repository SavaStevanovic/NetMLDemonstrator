from torch.utils.data import Dataset
import abc

class ClassDataset(Dataset, abc.ABC):

    @property
    @abc.abstractmethod
    def classes_map(self):
        pass