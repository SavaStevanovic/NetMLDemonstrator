import torch.optim as optim
import torch.nn as nn
import torch
import os
from model_fitting.losses import YoloLoss
from model_fitting.metrics import metrics
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torchvision
import random

def fit_epoch(net, trainloader, writer, lr_rate, box_transform, epoch=1):
    net.train()
    optimizer = optim.Adam(net.parameters(), lr_rate)
    criterion = YoloLoss(classes_len = net.classes, ratios = net.ratios)
    loss = 0.0
    objectness_f1s = 0.0
    for i, data in enumerate(tqdm(trainloader)):
        # get the inputs; data is a list of [inputs, labels]
        image, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(image.cuda())
        loss, objectness_f1 = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()

        loss += loss.item()
        objectness_f1s +=objectness_f1

    return loss/len(trainloader), objectness_f1s/len(trainloader)

def fit(net, trainloader, validationloader, chp_prefix, box_transform, epochs=1000, lower_learning_period=10):
    log_datatime = str(datetime.now().time())
    writer = SummaryWriter(os.path.join('logs', log_datatime))
    best_map = 0
    i = 0
    lr_rate = 0.0001
    for epoch in range(epochs):
        loss, objectness_f1 = fit_epoch(net, trainloader, writer, lr_rate, box_transform, epoch=epoch)
        train_map, train_samples = metrics(net, trainloader, box_transform, epoch)
        validation_map, validation_samples = metrics(net, validationloader, box_transform, epoch)
        writer.add_scalars('metrics', {'train_map':train_map, 'validation_map':validation_map, 'train_loss':loss, 'objectness_f1':objectness_f1}, epoch)
        for sample in train_samples:
            writer.add_images('train_sample', sample, epoch, dataformats='HWC')
        for sample in validation_samples:
            writer.add_images('validation_sample', sample, epoch, dataformats='HWC')
        if best_map < validation_map:
            i=0
            best_map = validation_map
            print('Epoch {}. Saving model with mAP: {}'.format(epoch, validation_map))
            chp_dir = 'checkpoints'
            os.makedirs((chp_dir), exist_ok=True)
            torch.save(net, os.path.join(chp_dir, '{}_checkpoints.pth'.format(chp_prefix)))
            with open(os.path.join(chp_dir, '{}_acc.txt'.format(chp_prefix)), 'w', encoding = 'utf-8') as f:
                f.write(str(best_map))
        else:
            i+=1
            print('Epoch {} mAP: {}'.format(epoch, validation_map))
        if i==lower_learning_period:
            lr_rate*=0.5
            i=0
            print("Learning rate lowered to {}".format(lr_rate))
    print('Finished Training')
    return best_map