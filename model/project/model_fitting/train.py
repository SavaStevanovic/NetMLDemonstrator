import torch.optim as optim
import torch.nn as nn
import torch
import os
from model_fitting.losses import YoloLoss
from model_fitting.metrics import metrics, validation_metrics
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

    for i, data in enumerate(tqdm(trainloader)):

        # get the inputs; data is a list of [inputs, labels]
        image, labels = data

        # pilImage = torchvision.transforms.ToPILImage()(image[0,...])
        # draw = ImageDraw.Draw(pilImage)
        # boxes = box_transform(labels)
        # for l in boxes:
        #     bbox = l['bbox']
        #     draw.rectangle(((bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])))
        #     draw.text((bbox[0], bbox[1]), trainloader.cats[l['category_id']][1])
        # if len(boxes):
        #     pilImage.save(os.path.join('demo_data', str(random.random())+'.png'), "png")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(image.cuda())
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()

        loss += loss.item()

    writer.add_scalar('training loss', loss/len(trainloader), epoch)

def fit(net, trainloader, validationloader, chp_prefix, box_transform, epochs=1000, lower_learning_period=10):
    log_datatime = str(datetime.now().time())
    writer = SummaryWriter(os.path.join('logs', log_datatime))
    best_acc = 0
    i = 0
    lr_rate = 0.001
    for epoch in range(epochs):
        fit_epoch(net, trainloader, writer, lr_rate, box_transform, epoch=epoch)
        train_f1_score, train_acc = metrics(net, trainloader, epoch)
        val_acc = validation_metrics(net, validationloader, epoch)
        writer.add_scalars('metrics', {'val_acc':val_acc, 'train_acc':train_acc, 'train_f1_score':train_f1_score}, epoch)
        if best_acc < val_acc:
            i=0
            best_acc = val_acc
            print('Epoch {}. Saving model with acc: {}'.format(epoch, val_acc))
            chp_dir = 'checkpoints'
            os.makedirs((chp_dir), exist_ok=True)
            torch.save(net, os.path.join(chp_dir, '{}_checkpoints.pth'.format(chp_prefix)))
            with open(os.path.join(chp_dir, '{}_acc.txt'.format(chp_prefix)), 'w', encoding = 'utf-8') as f:
                f.write(str(best_acc))
        else:
            i+=1
            print('Epoch {} acc: {}'.format(epoch, val_acc))
        if i==lower_learning_period:
            lr_rate*=0.5
            i=0
            print("Learning rate lowered to {}".format(lr_rate))
    print('Finished Training')
    return best_acc