from model import blocks
from model import networks
from data_loader.unified_dataloader import UnifiedKeypointDataloader
from model_fitting.train import fit
from torch import nn

th_count = 24
block_counts = [3, 4, 6]
depth = len(block_counts) + 2

dataloader = UnifiedKeypointDataloader(batch_size=16, depth=depth, th_count=th_count)

net = networks.Unet(
    block=blocks.ConvBlock,
    inplanes=64,
    in_dim=3,
    labels=dataloader.labels,
    norm_layer=nn.BatchNorm2d,
)

fit(
    net,
    dataloader.trainloader,
    dataloader.validationloader,
    epochs=1000,
    lower_learning_period=3,
)
