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
from functools import reduce
from torchsummary import summary

def fit_epoch(net, dataloader, lr_rate, box_transform, epoch=1):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    criterion = YoloLoss(ranges = net.ranges)
    losses = 0.0
    total_objectness_loss = 0.0
    total_size_loss = 0.0
    total_offset_loss = 0.0
    total_class_loss = 0.0
    images = []
    for i, data in enumerate(tqdm(dataloader)):
        image, labels = data
        optimizer.zero_grad()
        outputs = net(image.cuda())
        criterions = [criterion(outputs[i], labels[i].cuda()) for i in range(len(outputs))]
        loss, objectness_loss, size_loss, offset_loss, class_loss = (sum(x) for x in zip(*criterions))
        loss.backward()
        optimizer.step()
        total_objectness_loss += objectness_loss
        total_size_loss += size_loss
        losses += loss.item()
        total_offset_loss += offset_loss
        total_class_loss += class_loss

        if i>=len(dataloader)-5:
            outs = [out.detach().cpu()[0].numpy() for out in outputs]
            labs = [labels[0].cpu()[0].numpy()]
            pilImage = apply_detections(box_transform, outs, labs, image[0], dataloader.cats)
            images.append(pilImage)
        
    data_len = len(dataloader)
    return losses/data_len, total_objectness_loss/data_len, total_size_loss/data_len, total_offset_loss/data_len, total_class_loss/data_len, images

def fit(net, trainloader, validationloader, dataset_name, box_transform, epochs=1000, lower_learning_period=10):
    model_dir_header = net.get_identifier()
    chp_dir = os.path.join('checkpoints', model_dir_header)
    checkpoint_name_path = os.path.join(chp_dir, '{}_checkpoints.pth'.format(dataset_name))
    checkpoint_conf_path = os.path.join(chp_dir, '{}_configuration.json'.format(dataset_name))
    train_config = TrainingConfiguration()
    if os.path.exists(chp_dir):
        net = torch.load(checkpoint_name_path)
        train_config.load(checkpoint_conf_path)
    net.cuda()
    summary(net, (3, 224, 224))
    writer = SummaryWriter(os.path.join('logs', model_dir_header))
    for epoch in range(train_config.epoch, epochs):
        loss, objectness_loss, size_loss, offset_loss, class_loss, samples = fit_epoch(net, trainloader, train_config.learning_rate, box_transform, epoch=epoch)
        writer.add_scalars('Train/Metrics', {'objectness_loss': objectness_loss, 'size_loss':size_loss, 'offset_loss':offset_loss, 'class_loss':class_loss}, epoch)
        writer.add_scalar('Train/Metrics/loss', loss, epoch)
        grid = join_images(samples)
        writer.add_images('train_sample', grid, epoch, dataformats='HWC')
        
        validation_map, loss, objectness_loss, size_loss, offset_loss, class_loss, samples = metrics(net, validationloader, box_transform, epoch)
        writer.add_scalars('Validation/Metrics', {'objectness_loss': objectness_loss, 'size_loss':size_loss, 'offset_loss':offset_loss, 'class_loss':class_loss}, epoch)
        writer.add_scalar('Validation/Metrics/loss', loss, epoch)
        writer.add_scalar('Validation/Metrics/validation_map', validation_map, epoch)
        grid = join_images(samples)
        writer.add_images('validation_sample', grid, epoch, dataformats='HWC')

        os.makedirs((chp_dir), exist_ok=True)
        if train_config.best_metric > loss:
            train_config.iteration_age = 0
            train_config.best_metric = loss
            print('Epoch {}. Saving model with metric: {}'.format(epoch, loss))
            torch.save(net, checkpoint_name_path.replace('.pth', '_final.pth'))
        else:
            train_config.iteration_age+=1
            print('Epoch {} metric: {}'.format(epoch, loss))
        if train_config.iteration_age==lower_learning_period:
            train_config.learning_rate*=0.5
            train_config.iteration_age=0
            print("Learning rate lowered to {}".format(train_config.learning_rate))
        train_config.epoch = epoch+1
        train_config.save(checkpoint_conf_path)
        torch.save(net, checkpoint_name_path)
    print('Finished Training')
    return best_map