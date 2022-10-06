import torch
from data_loader.unified_dataloader import UnifiedDataloader
from data_loader.unified_dataset import DetectionDatasetWrapper, UnifiedDataset
from model import blocks
from model import networks
from model_fitting.train import fit

torch.backends.cudnn.benchmark = True

th_count = 24
ratios = [0.5, 1.0, 2.0]
dataset_name = 'Coco'
block_size = [3, 4, 6, 3]
batch_size = 8
backbone = networks.ResNetBackbone(
    inplanes=64, block=blocks.BasicBlock, block_counts=block_size)
train_dataset = UnifiedDataset(True, len(block_size)+1, debug=th_count)
val_dataset = UnifiedDataset(False, len(block_size)+1, debug=th_count)
net = networks.YoloV2(classes=val_dataset.classes_map, ratios=ratios)

train_dataset = DetectionDatasetWrapper(train_dataset, net)
val_dataset = DetectionDatasetWrapper(val_dataset, net)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=(
    th_count > 1)*(batch_size-1)+1, shuffle=th_count > 1, num_workers=th_count)
validationloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=th_count//2)


fit(net, 
    trainloader, 
    validationloader,
    dataset_name=dataset_name,
    epochs=1000,
    lower_learning_period=5)
