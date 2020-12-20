from torch import nn
from model import utils
import itertools
import functools
import torch
import numpy as np

      
class OpenPoseNet(nn.Module, utils.Identifier):
    def __init__(self, backbone, paf_stages, map_stages, block, block_count, paf_planes, map_planes):
        super(OpenPoseNet, self).__init__()
        self.backbone = functools.reduce(lambda b,m : m(b),backbone[::-1])
        self.block = block
        self.paf_stages = paf_stages
        self.map_stages = map_stages
        self.paf_planes = paf_planes
        self.map_planes = map_planes
        self.adapter = nn.Sequential(
            nn.Conv2d(self.backbone.channels   , self.backbone.channels//2, kernel_size=1, bias=True, padding=0),
            nn.PReLU(),
            nn.Conv2d(self.backbone.channels//2, self.backbone.channels//4, kernel_size=1, bias=True, padding=0),
        )
        self.channels = self.backbone.channels//4
        self.first_paf = block(self.channels, self.channels, paf_planes, block_count)
        self.pafs = nn.ModuleList([block(self.channels + paf_planes, self.channels, paf_planes, block_count) for _ in range(paf_stages-1)])
        self.first_map = block(self.channels + paf_planes, self.channels, map_planes, block_count)
        self.maps = nn.ModuleList([block(self.channels + map_planes, self.channels, map_planes, block_count) for _ in range(map_stages-1)])
    
    def forward(self, x):
        o_size = x.size()[2:][::-1]
        feature = self.adapter(self.backbone(x))
        out = self.first_paf(feature)
        pafs_features = [out]
        out = torch.cat([out, feature], 1)
        for paf in self.pafs:
            out = paf(out)
            pafs_features.append(out)
            out = torch.cat([out, feature], 1)

        out = self.first_map(out)
        maps_features = [out]
        for mapf in self.maps:
            out = torch.cat([out, feature], 1)
            out = mapf(out)
            maps_features.append(out)
        
        return tuple(pafs_features), tuple(maps_features)

class VGGNetBackbone(nn.Sequential, utils.Identifier):
    def __init__(self, inplanes, block_counts):
        self.channels = inplanes
        self.inplanes = inplanes
        self.block_counts = block_counts
        layers = [nn.Conv2d(3, self.channels, kernel_size=3, bias=True, padding=1)]
        for _ in range(self.block_counts[0]-1):
            layers.extend([nn.PReLU(), nn.Conv2d(self.channels, self.channels, kernel_size=3, bias=True, padding=1)])
        for b in self.block_counts[1:]:
            layers.append(nn.MaxPool2d(2, 2))
            layers.extend([nn.PReLU(), nn.Conv2d(self.channels, self.channels*2, kernel_size=3, bias=True, padding=1)])
            self.channels*=2
            for _ in range(b-1):
                layers.extend([nn.PReLU(), nn.Conv2d(self.channels, self.channels, kernel_size=3, bias=True, padding=1)])
           
        super(VGGNetBackbone, self).__init__(*layers)


class CocoPoseNet(nn.Module, utils.Identifier):
    insize = 416
    def __init__(self, path = None):
        super(CocoPoseNet, self).__init__()
        self.base = Base_model()
        self.stage_1 = Stage_1()
        self.stage_2 = Stage_x()
        self.stage_3 = Stage_x()
        self.stage_4 = Stage_x()
        self.stage_5 = Stage_x()
        self.stage_6 = Stage_x()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, 0)
        if path:
            self.base.vgg_base.load_state_dict(torch.load(path))
        
    def forward(self, x):
        heatmaps = []
        pafs = []
        feature_map = self.base(x)
        h1, h2 = self.stage_1(feature_map)
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_2(torch.cat([h1, h2, feature_map], dim = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_3(torch.cat([h1, h2, feature_map], dim = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_4(torch.cat([h1, h2, feature_map], dim = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_5(torch.cat([h1, h2, feature_map], dim = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_6(torch.cat([h1, h2, feature_map], dim = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        return pafs, heatmaps     

class VGG_Base(nn.Module):
    def __init__(self):
        super(VGG_Base, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv1_2 = nn.Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, stride = 1, padding = 1)
        self.conv2_1 = nn.Conv2d(in_channels = 64, out_channels = 128,  kernel_size = 3, stride = 1, padding = 1)
        self.conv2_2 = nn.Conv2d(in_channels = 128, out_channels = 128,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_1 = nn.Conv2d(in_channels = 128, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_2 = nn.Conv2d(in_channels = 256, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_3 = nn.Conv2d(in_channels = 256, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_4 = nn.Conv2d(in_channels = 256, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv4_1 = nn.Conv2d(in_channels = 256, out_channels = 512,  kernel_size = 3, stride = 1, padding = 1)
        self.conv4_2 = nn.Conv2d(in_channels = 512, out_channels = 512,  kernel_size = 3, stride = 1, padding = 1)
        self.relu = nn.PReLU()
        self.max_pooling_2d = nn.MaxPool2d(kernel_size = 2, stride = 2)
    
    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.conv3_4(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        return x

class Base_model(nn.Module):
    def __init__(self):
        super(Base_model, self).__init__()
        self.vgg_base = VGG_Base()
        self.conv4_3_CPM = nn.Conv2d(in_channels=512, out_channels=256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv4_4_CPM = nn.Conv2d(in_channels=256, out_channels=128,  kernel_size = 3, stride = 1, padding = 1)
        self.relu = nn.PReLU()
    def forward(self, x):
        x = self.vgg_base(x)
        x = self.relu(self.conv4_3_CPM(x))
        x = self.relu(self.conv4_4_CPM(x))
        return x
    
class Stage_1(nn.Module):
    def __init__(self):
        super(Stage_1, self).__init__()
        self.conv1_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L1 = nn.Conv2d(in_channels=512, out_channels=38, kernel_size=1, stride=1, padding=0)
        self.conv1_CPM_L2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_CPM_L2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_CPM_L2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_CPM_L2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L2 = nn.Conv2d(in_channels=512, out_channels=18, kernel_size=1, stride=1, padding=0)
        self.relu = nn.PReLU()
        
    def forward(self, x):
        h1 = self.relu(self.conv1_CPM_L1(x)) # branch1
        h1 = self.relu(self.conv2_CPM_L1(h1))
        h1 = self.relu(self.conv3_CPM_L1(h1))
        h1 = self.relu(self.conv4_CPM_L1(h1))
        h1 = self.conv5_CPM_L1(h1)
        h2 = self.relu(self.conv1_CPM_L2(x)) # branch2
        h2 = self.relu(self.conv2_CPM_L2(h2))
        h2 = self.relu(self.conv3_CPM_L2(h2))
        h2 = self.relu(self.conv4_CPM_L2(h2))
        h2 = self.conv5_CPM_L2(h2)
        return h1, h2
    
class Stage_x(nn.Module):
    def __init__(self):
        super(Stage_x, self).__init__()
        self.conv1_L1 = nn.Conv2d(in_channels = 184, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv2_L1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv3_L1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv4_L1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv5_L1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv6_L1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv7_L1 = nn.Conv2d(in_channels = 128, out_channels = 38, kernel_size = 1, stride = 1, padding = 0)
        self.conv1_L2 = nn.Conv2d(in_channels = 184, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv2_L2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv3_L2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv4_L2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv5_L2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv6_L2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv7_L2 = nn.Conv2d(in_channels = 128, out_channels = 18, kernel_size = 1, stride = 1, padding = 0)
        self.relu = nn.PReLU()
        
    def forward(self, x):
        h1 = self.relu(self.conv1_L1(x)) # branch1
        h1 = self.relu(self.conv2_L1(h1))
        h1 = self.relu(self.conv3_L1(h1))
        h1 = self.relu(self.conv4_L1(h1))
        h1 = self.relu(self.conv5_L1(h1))
        h1 = self.relu(self.conv6_L1(h1))
        h1 = self.conv7_L1(h1)
        h2 = self.relu(self.conv1_L2(x)) # branch2
        h2 = self.relu(self.conv2_L2(h2))
        h2 = self.relu(self.conv3_L2(h2))
        h2 = self.relu(self.conv4_L2(h2))
        h2 = self.relu(self.conv5_L2(h2))
        h2 = self.relu(self.conv6_L2(h2))
        h2 = self.conv7_L2(h2)
        return h1, h2