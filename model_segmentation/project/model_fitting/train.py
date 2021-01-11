import torch
import os
from model_fitting.losses import SegmentationLoss
from sklearn.metrics import f1_score
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
import torch.nn.functional as F
from utils import plt_to_np

def focal_loss(x, y):
    gamma = 2
    alpha = 0.25

    y = y.unsqueeze(-1)
    x = x.unsqueeze(-1)

    y = torch.cat([(1-alpha) * y, alpha * (1-y)], -1)
    x = torch.cat([x, 1-x], -1).clamp(1e-8, 1. - 1e-8)

    F_loss = -y * (1 - x)**gamma * torch.log(x)

    return F_loss.mean()

def fit_epoch(net, dataloader, lr_rate, train, epoch=1):
    if train:
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    else:
        net.eval()
    criterion = SegmentationLoss()
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    losses = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    f1_metric = 0.0
    output_images = []
    label_images = []

    for i, data in enumerate(tqdm(dataloader)):
        image, labels = data
        if train:
            optimizer.zero_grad()
        output = net(image.cuda())
        labels_cuda = labels.cuda()
        loss, focal_loss, dice_loss = criterion(output, labels_cuda)
        if train:
            loss.backward()
            optimizer.step()
        losses += loss.item()
        total_focal_loss += focal_loss
        total_dice_loss += dice_loss
        if not train:
            outputs = output.detach().softmax(1).cpu().numpy()
            output_threshold = (outputs[:, 1] > 0.5).astype('float')
            f1_metric += f1_score(output_threshold.flatten(), labels.argmax(1).flatten(), average='weighted')

        if i>=len(dataloader)-5:
            image = image[0].permute(1,2,0).detach().cpu().numpy()
            label = labels[0].detach().cpu().numpy()
            output = output[0].detach().cpu().numpy()
        

            plt.imshow(image)
            plt.imshow(label.argmax(axis=0), alpha=0.7)
            label_images.append(plt_to_np(plt))


            plt.imshow(image)
            plt.imshow(output.argmax(axis=0), alpha=0.7)
            output_images.append(plt_to_np(plt))

    data_len = len(dataloader)
    return losses/data_len, output_images, label_images, total_focal_loss/data_len, total_dice_loss/data_len, f1_metric/data_len

def fit(net, trainloader, validationloader, epochs=1000, lower_learning_period=10):
    model_dir_header = net.get_identifier()
    chp_dir = os.path.join('checkpoints', model_dir_header)
    checkpoint_name_path = os.path.join(chp_dir, 'checkpoints.pth')
    checkpoint_conf_path = os.path.join(chp_dir, 'configuration.json')
    train_config = TrainingConfiguration()
    if os.path.exists(chp_dir):
        net = torch.load(checkpoint_name_path)
        train_config.load(checkpoint_conf_path)
    net.cuda()
    summary(net, (3, 448, 448))
    writer = SummaryWriter(os.path.join('logs', model_dir_header))
    images, _ = next(iter(trainloader))

    with torch.no_grad():
        writer.add_graph(net, images[:2].cuda())
    for epoch in range(train_config.epoch, epochs):
        loss, output_images, label_images, focal_loss, dice_loss, f1_metric = fit_epoch(net, trainloader, train_config.learning_rate, train = True, epoch=epoch)
        writer.add_scalars('Train/Metrics', {'loss': loss, 'focal_loss': focal_loss, 'dice_loss': dice_loss, 'f1_metric': f1_metric}, epoch)
        grid = join_images(label_images)
        writer.add_images('train_labels', grid, epoch, dataformats='HWC')
        grid = join_images(output_images)
        writer.add_images('train_outputs', grid, epoch, dataformats='HWC')

        loss, output_images, label_images, focal_loss, dice_loss, f1_metric = fit_epoch(net, validationloader, None, train = False, epoch=epoch)
        writer.add_scalars('Validation/Metrics', {'loss': loss, 'focal_loss': focal_loss, 'dice_loss': dice_loss, 'f1_metric': f1_metric}, epoch)
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