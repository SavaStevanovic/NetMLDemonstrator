from data_loader import augmentation
from data_loader.manual_dataset import ManualDetection

from data_loader.indoors_dataset import IndoorDetection
from data_loader.class_dataset import ClassDataset
from data_loader.subset_dataset import SubsetDataset
from data_loader.voc_dataset import VOCDataset
from visualization.output_transform import TargetTransform


class UnifiedDataset(ClassDataset):
    def __init__(self, train, depth, debug=False):
        self.debug = debug
        self.train = train
        val_data = IndoorDetection(
                    "/Data/detection/IndoorObjectDetectionDataset/validation")
        train_datasets = [
            # IndoorDetection(
            #     "/Data/detection/IndoorObjectDetectionDataset/train"),
            # ManualDetection("/Data/detection/manual/voc"),
            SubsetDataset(VOCDataset(mode="train", directory="/Data/detection/VOC/"), {"tvmonitor": "screen", 
                                                                                    #    "chair": "chair"
                                                                                       })
        ]

        if train:
            self.datasets = train_datasets
        if not train:
            self.datasets = [
                # VOCDataset('val', 'Voc'),
                # ADEChallengeData2016('val', 'ADEChallengeData2016'),
                # CityscapesDataset('val', 'fine', 'Cityscapes'),
                val_data,
            ]

        if train:
            self.transforms = augmentation.PairCompose([
                augmentation.RandomResizeTransform(),
                augmentation.RandomHorizontalFlipTransform(),
                augmentation.RandomCropTransform((416, 416)),
                augmentation.RandomNoiseTransform(),
                augmentation.RandomColorJitterTransform(),
                augmentation.RandomBlurTransform(),
                augmentation.RandomJPEGcompression(95),
                augmentation.OutputTransform()]
            )
        if not train:
            self.transforms = augmentation.PairCompose([
                augmentation.PaddTransform(pad_size=2**depth),
                augmentation.OutputTransform()]
            )

        if self.debug == 1:
            self.data_ids = [(i, j) for i, dataset in enumerate(
                self.datasets) for j in range(20)]
        else:
            self.data_ids = [(i, j) for i, dataset in enumerate(
                self.datasets) for j in range(len(self.datasets[i]))]

    @property
    def classes_map(self):
        return sorted(set(sum([x.classes_map for x in self.datasets], [])))
    
    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        identifier = self.data_ids[idx]
        dataset = self.datasets[identifier[0]]
        data, labels = dataset[identifier[1]]
        if self.transforms:
            data = self.transforms(data, labels)
        return data, tuple(dataset.classes_map)


class DetectionDatasetWrapper(ClassDataset):
    def __init__(self, dateset: ClassDataset, net) -> None:
        super().__init__()
        self._dateset = dateset
        self._output_transformation = TargetTransform(prior_box_sizes=net.prior_box_sizes, classes=net.classes, ratios=net.ratios, strides=net.strides)
        
    @property
    def classes_map(self):
        self._dateset.classes_map    

    def __len__(self):
        return len(self._dateset)
    
    def __getitem__(self, idx):
        data, labels = self._dateset.__getitem__(idx)
        return self._output_transformation(*data), "%".join(labels)