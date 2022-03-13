from model import blocks
from model import networks
from data_loader.unified_dataloader import UnifiedKeypointDataloader
from model_fitting.train import fit

net = networks.FCN(3, 3)

dataloader = UnifiedKeypointDataloader(
    batch_size=4, depth=net.depth, th_count=24)


fit(
    net,
    dataloader.trainloader,
    dataloader.validationloader,
    epochs=1000,
    lower_learning_period=3
)
