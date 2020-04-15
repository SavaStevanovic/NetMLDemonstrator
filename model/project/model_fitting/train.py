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
import numpy as np

def fit_epoch(net, dataloader, writer, lr_rate, box_transform, epoch=1):
    net.train()
    optimizer = optim.Adam(net.parameters(), lr_rate)
    criterion = YoloLoss(classes_len = net.classes, ratios = net.ratios)
    losses = 0.0
    objectness_f1s = 0.0
    total_objectness_loss = 0.0
    total_size_loss = 0.0
    total_offset_loss = 0.0
    total_class_loss = 0
    images = []
    for i, data in enumerate(tqdm(dataloader)):
        # get the inputs; data is a list of [inputs, labels]
        image, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(image.cuda())
        loss, objectness_f1, objectness_loss, size_loss, offset_loss, class_loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()
        total_objectness_loss += objectness_loss.item()
        total_size_loss += size_loss.item()
        losses += loss.item()
        total_offset_loss += offset_loss.item()
        total_class_loss += class_loss.item()
        objectness_f1s +=objectness_f1

        if i>len(dataloader)-5:
            object_range = 5*len(net.ratios)+net.classes
            outputs[:, ::object_range] = torch.sigmoid(outputs[:, ::object_range])

            offset_range = [3,4]
            box_offset_range = [i for i in range(labels.shape[1]) if i%object_range in offset_range]
            outputs[:,box_offset_range] = torch.sigmoid(outputs[:,box_offset_range])

            boxes_pr = box_transform(outputs.cpu().detach())
            boxes_tr = box_transform(labels.cpu().detach())
            pilImage = torchvision.transforms.ToPILImage()(image[0,...])
            draw = ImageDraw.Draw(pilImage)
            for l in boxes_pr:
                bbox = l['bbox']
                draw.rectangle(((bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])), outline = 'red')
                draw.text((bbox[0], bbox[1]-10), dataloader.cats[l['category_id']][1], outline = 'red')
            for l in boxes_tr:
                bbox = l['bbox']
                draw.rectangle(((bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])), outline = 'blue')
                draw.text((bbox[0], bbox[1]-10), dataloader.cats[l['category_id']][1], outline = 'blue')
            images.append(np.array(pilImage))

    data_len = len(dataloader)
    return losses/data_len, objectness_f1s/data_len, total_objectness_loss/data_len, total_size_loss/data_len, total_offset_loss/data_len, total_class_loss/data_len, images

def fit(net, trainloader, validationloader, chp_prefix, box_transform, epochs=1000, lower_learning_period=10):
    log_datatime = str(datetime.now().time())
    writer = SummaryWriter(os.path.join('logs', log_datatime))
    best_map = 0
    i = 0
    lr_rate = 0.0001
    for epoch in range(epochs):
        loss, objectness_f1, objectness_loss, size_loss, offset_loss, class_loss, train_samples = fit_epoch(net, trainloader, writer, lr_rate, box_transform, epoch=epoch)
        # train_map, train_samples = metrics(net, trainloader, box_transform, epoch)
        validation_map, validation_samples = metrics(net, validationloader, box_transform, epoch)
        writer.add_scalars('metrics', {'validation_map':validation_map, 'train_loss':loss, 'objectness_f1':objectness_f1, 'objectness_loss': objectness_loss, 'size_loss':size_loss, 'offset_loss':offset_loss, 'class_loss':class_loss}, epoch)
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