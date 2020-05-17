import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Identifier(object):
    def __init__(self):
        pass

    def get_identifier(self):
        idt = self.__class__.__name__
        if hasattr(self, 'inplanes'): 
            idt+='/'+str(self.inplanes)
        if hasattr(self, 'block_counts'): 
            idt+='/'+'-'.join([str(x) for x in self.block_counts])
        if hasattr(self, 'ratios'): 
            idt+='/'+'-'.join([str(x).replace('.',',') for x in self.ratios])
        if hasattr(self, 'block') and hasattr(self.block, 'get_identifier'): 
            idt+='/'+self.block.__name__
        if hasattr(self, 'backbone') and hasattr(self.backbone, 'get_identifier'): 
            idt+='/'+self.backbone.get_identifier() 
        return idt
