import torch
import os
from model_fitting.losses import YoloLoss
from model_fitting.metrics import metrics
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from visualization.display_detection import apply_detections
from visualization.images_display import join_images

def fit_epoch(net, dataloader, writer, lr_rate, box_transform, epoch=1):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    criterion = YoloLoss(ranges = net.ranges)
    losses = 0.0
    objectness_f1s = 0.0
    total_objectness_loss = 0.0
    total_size_loss = 0.0
    total_offset_loss = 0.0
    total_class_loss = 0
    images = []
    for i, data in enumerate(tqdm(dataloader)):
        image, labels = data
        optimizer.zero_grad()
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
            pilImage = apply_detections(box_transform, outputs.cpu().detach(), labels.cpu().detach(), image[0,...], dataloader.cats)
            images.append(pilImage)

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
        validation_map, validation_samples = metrics(net, validationloader, box_transform, epoch)
        writer.add_scalars('metrics', {'validation_map':validation_map, 'train_loss':loss, 'objectness_f1':objectness_f1, 'objectness_loss': objectness_loss, 'size_loss':size_loss, 'offset_loss':offset_loss, 'class_loss':class_loss}, epoch)
        grid = join_images(train_samples)
        writer.add_images('train_sample', grid, epoch, dataformats='HWC')
        grid = join_images(validation_samples)
        writer.add_images('validation_sample', grid, epoch, dataformats='HWC')
        if best_map < validation_map:
            i=0
            best_map = validation_map
            print('Epoch {}. Saving model with mAP: {}'.format(epoch, validation_map))
            chp_dir = 'checkpoints'
            os.makedirs((chp_dir), exist_ok=True)
            torch.save(net, os.path.join(chp_dir, '{}_checkpoints.pth'.format(chp_prefix)))
            with open(os.path.join(chp_dir, '{}_metric.txt'.format(chp_prefix)), 'w', encoding = 'utf-8') as f:
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