from model import blocks
from model import networks
from data_loader.unified_dataloader import UnifiedKeypointDataloader
from model_fitting.train import fit
from torch import nn

th_count = 1
block_counts = [3, 4, 6]
depth = 5

dataloaders = UnifiedKeypointDataloader(batch_size=6, depth=depth, th_count=th_count)

train_data, val_data = next(iter(dataloaders))
net = networks.Resnet(
    block=blocks.ConvBlock,
    inplanes=64,
    in_dim=3,
    depth=5,
    labels=dataloaders.labels,  
    norm_layer=nn.BatchNorm2d,
)

fit(net, train_data, val_data, epochs=1000, lower_learning_period=3, split=0)
