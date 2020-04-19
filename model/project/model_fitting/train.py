import torch
import os
from model_fitting.losses import YoloLoss
from model_fitting.metrics import metrics
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from visualization.display_detection import apply_detections
from visualization.images_display import join_images
from model_fitting.configuration import TrainingConfiguration
import json

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
        objectness_f1s +=objectness_f1.item()

        if i>len(dataloader)-5:
            pilImage = apply_detections(box_transform, outputs.cpu().detach(), labels.cpu().detach(), image[0,...], dataloader.cats)
            images.append(pilImage)
        
    data_len = len(dataloader)
    return losses/data_len, objectness_f1s/data_len, total_objectness_loss/data_len, total_size_loss/data_len, total_offset_loss/data_len, total_class_loss/data_len, images

def fit(net, trainloader, validationloader, dataset_name, box_transform, epochs=1000, lower_learning_period=10):
    model_dir_header = net.get_identifier()
    chp_dir = os.path.join('checkpoints', model_dir_header)
    checkpoint_name_path = os.path.join(chp_dir, '{}_checkpoints.pth'.format(dataset_name))
    checkpoint_conf_path = os.path.join(chp_dir, '{}_configuration.json'.format(dataset_name))
    train_config = TrainingConfiguration()
    if os.path.exists(chp_dir):
        net = torch.load(checkpoint_name_path)
        train_config.load(checkpoint_conf_path)
    writer = SummaryWriter(os.path.join('logs', model_dir_header))
    for epoch in range(train_config.epoch, epochs):
        train_config.epoch = epoch+1
        loss, objectness_f1, objectness_loss, size_loss, offset_loss, class_loss, train_samples = fit_epoch(net, trainloader, writer, train_config.learning_rate, box_transform, epoch=epoch)
        validation_map, validation_samples = metrics(net, validationloader, box_transform, epoch)
        writer.add_scalars('Train/Metrics', {'objectness_loss': objectness_loss, 'size_loss':size_loss, 'offset_loss':offset_loss, 'class_loss':class_loss}, epoch)
        writer.add_scalar('Train/Metrics/loss', loss, epoch)
        writer.add_scalar('Train/Metrics/objectness_f1', objectness_f1, epoch)
        writer.add_scalar('Validation/Metrics/validation_map', validation_map, epoch)

        grid = join_images(train_samples)
        writer.add_images('train_sample', grid, epoch, dataformats='HWC')
        grid = join_images(validation_samples)
        writer.add_images('validation_sample', grid, epoch, dataformats='HWC')
        os.makedirs((chp_dir), exist_ok=True)
        if train_config.best_metric < validation_map:
            train_config.iteration_age = 0
            train_config.best_metric = validation_map
            print('Epoch {}. Saving model with mAP: {}'.format(epoch, validation_map))
            torch.save(net, checkpoint_name_path)
        else:
            train_config.iteration_age+=1
            print('Epoch {} mAP: {}'.format(epoch, validation_map))
        if train_config.iteration_age==lower_learning_period:
            train_config.learning_rate*=0.5
            train_config.iteration_age=0
            print("Learning rate lowered to {}".format(train_config.learning_rate))
        train_config.save(checkpoint_conf_path)
    print('Finished Training')
    return best_map