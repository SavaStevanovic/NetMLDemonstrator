import torch
import os
from model_fitting.metrics import metrics
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from visualization.apply_output import apply_detections
from visualization.images_display import join_images
from model_fitting.configuration import TrainingConfiguration
import json
from functools import reduce
from torchsummary import summary
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np
from visualization.output_transform import get_mapped_image 
from PIL import Image


def fit_epoch(net, dataloader, lr_rate, postprocessing, epoch=1):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    criterion = torch.nn.MSELoss()
    losses = 0.0
    total_pafs_loss = 0.0
    total_maps_loss = 0.0
    output_images = []
    label_images = []
    for i, data in enumerate(tqdm(dataloader)):
        image, labels = data
        # labels_cuda = labels.cuda()
        optimizer.zero_grad()
        pafs_output, maps_output = net(image.cuda())
        paf_label = labels[0].cuda()
        map_label = labels[1].cuda()
        pafs_loss = torch.stack([(paf_label>0).int() * criterion(paf_label, o) for o in pafs_output], dim=0).sum()
        maps_loss = torch.stack([(map_label>0).int() * criterion(map_label, o) for o in maps_output], dim=0).sum()
        loss = pafs_loss + maps_loss
        loss.backward()
        optimizer.step()
        total_pafs_loss += pafs_loss.item()
        total_maps_loss += maps_loss.item()
        losses += loss.item()

        if i>=len(dataloader)-5:
            mapped_image = get_mapped_image(image, labels, postprocessing, dataloader.skeleton, dataloader.parts)
            label_images.append(np.array(mapped_image))

            mapped_image = get_mapped_image(image, [pafs_output[-1], maps_output[-1]], postprocessing, dataloader.skeleton, dataloader.parts)
            output_images.append(np.array(mapped_image))

    data_len = len(dataloader)
    return losses/data_len, total_pafs_loss/data_len, total_maps_loss/data_len, output_images, label_images


def fit(net, trainloader, validationloader, postprocessing, epochs=1000, lower_learning_period=10):
    model_dir_header = net.get_identifier()
    chp_dir = os.path.join('checkpoints', model_dir_header)
    checkpoint_name_path = os.path.join(chp_dir, 'checkpoints.pth')
    checkpoint_conf_path = os.path.join(chp_dir, 'configuration.json')
    train_config = TrainingConfiguration()
    if os.path.exists(chp_dir):
        net = torch.load(checkpoint_name_path)
        train_config.load(checkpoint_conf_path)
    net.cuda()
    summary(net, (3, 224, 224))
    writer = SummaryWriter(os.path.join('logs', model_dir_header))
    for epoch in range(train_config.epoch, epochs):
        loss, total_pafs_loss, total_maps_loss, output_images, label_images = fit_epoch(net, trainloader, train_config.learning_rate, postprocessing, epoch=epoch)
        writer.add_scalars('Train/Metrics', {'total_pafs_loss': total_pafs_loss, 'total_maps_loss': total_maps_loss}, epoch)
        writer.add_scalar('Train/Metrics/loss', loss, epoch)
        grid = join_images(label_images)
        writer.add_images('train_labels', grid/255, epoch, dataformats='HWC')
        grid = join_images(output_images)
        writer.add_images('train_outputs', grid/255, epoch, dataformats='HWC')

        loss, total_pafs_loss, total_maps_loss, output_images, label_images = metrics(net, validationloader, postprocessing, epoch=epoch)
        writer.add_scalars('Validation/Metrics', {'total_pafs_loss': total_pafs_loss, 'total_maps_loss': total_maps_loss}, epoch)
        writer.add_scalar('Validation/Metrics/loss', loss, epoch)
        grid = join_images(label_images)
        writer.add_images('validation_labels', grid, epoch, dataformats='HWC')
        grid = join_images(output_images)
        writer.add_images('validation_outputs', grid, epoch, dataformats='HWC')

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
        torch.save(net.state_dict(), checkpoint_name_path.replace('.pth', '_final_state_dict.pth'))
    print('Finished Training')
    return best_map