import torch
import os
from model_fitting.losses import SegmentationLoss
from sklearn.metrics import f1_score
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from visualization.apply_output import apply_detections
from visualization.images_display import join_images, visualize_label
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
from model_fitting import metrics
from operator import itemgetter

def fit_epoch(net, dataloader, lr_rate, train, epoch=1):
    if train:
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    else:
        net.eval()
        cm = metrics.RunningConfusionMatrix(range(net.out_dim))
    criterion = SegmentationLoss()
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    losses = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    output_images = []
    label_images = []
    outputs_vector = []
    label_vector = []
    f1_scores = 0
    accs = 0
    sel_len = len(set(sum(dataloader.selector,[])))

    for i, data in enumerate(tqdm(dataloader)):
        image, labels, dataset_id = data
        # print(image.shape)
        sel = np.zeros((len(dataset_id), sel_len))
        for j, x in enumerate(dataset_id):
            sel[j, dataloader.selector[x]]=1
        if train:
            optimizer.zero_grad()
        mask = 1 - labels[:, -1]
        labels_cuda = labels.cuda()
        labels_cuda = labels_cuda[:, 1:]
        mask_cuda = mask.cuda()
        labels_cuda = labels_cuda[:, :-1]
        outputs = net(image.cuda())
        outputs = outputs * mask_cuda.unsqueeze(1)
        loss, focal_loss, dice_loss = criterion(outputs, labels_cuda)
        if train:
            loss.backward()
            optimizer.step()
        losses += loss.item()
        total_focal_loss += focal_loss
        total_dice_loss += dice_loss
        if not train:
            flat_mask = mask_cuda.flatten().bool()
            lab = labels_cuda[:, dataloader.selector[dataset_id]]
            lab = torch.cat((1-lab.sum(1, keepdim=True), lab), 1).argmax(1).flatten()
                lab = lab[flat_mask]
            out = outputs[:, dataloader.selector[dataset_id]]
            out = torch.cat((1-out.sum(1, keepdim=True), out), 1).argmax(1).flatten()
                out = out[flat_mask]
            accs += lab.eq(out).float().mean().item()

        if i>=len(dataloader)-5:
            image = image[0].permute(1,2,0).detach().cpu().numpy()
            label = labels[0, 1:-1].detach().cpu()
            output = outputs[0].detach().cpu()
            lab = label[dataloader.selector[dataset_id[0]]]
            lab = torch.cat((1-lab.sum(0).unsqueeze(0), lab), 0).argmax(0).numpy()
            out = output.detach().cpu()[dataloader.selector[dataset_id[0]]]
            out = torch.cat((1-out.sum(0).unsqueeze(0), out), 0).argmax(0).numpy()
            color_map = list(itemgetter(*dataloader.selector[dataset_id[0]])(dataloader.color_set))
            color_map = [[0, 0, 0]] + color_map
            lab = visualize_label(color_map, lab)
            out = visualize_label(color_map, out)
            plt.imshow(image)
            plt.imshow(lab, alpha=0.75)
            label_images.append(plt_to_np(plt))


            plt.imshow(image)
            plt.imshow(out, alpha=0.75)
            output_images.append(plt_to_np(plt))

    # miou = cm.compute_current_mean_intersection_over_union()
    # if not train:
        # f1_scores = f1_score(np.concatenate(label_vector).flatten(), np.concatenate(outputs_vector).flatten(), average='macro')
    data_len = len(dataloader)
    return losses/data_len, output_images, label_images, total_focal_loss/data_len, total_dice_loss/data_len, accs/data_len

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
    images, _, _ = next(iter(trainloader))

    with torch.no_grad():
        writer.add_graph(net, images[:2].cuda())
    for epoch in range(train_config.epoch, epochs):
        loss, output_images, label_images, focal_loss, dice_loss, _ = fit_epoch(net, trainloader, train_config.learning_rate, train = True, epoch=epoch)
        writer.add_scalars('Train/Metrics', {'loss': loss, 'focal_loss': focal_loss, 'dice_loss': dice_loss}, epoch)
        grid = join_images(label_images)
        writer.add_images('train_labels', grid, epoch, dataformats='HWC')
        grid = join_images(output_images)
        writer.add_images('train_outputs', grid, epoch, dataformats='HWC')

        loss, output_images, label_images, focal_loss, dice_loss, accs = fit_epoch(net, validationloader, None, train = False, epoch=epoch)
        writer.add_scalars('Validation/Metrics', {'loss': loss, 'focal_loss': focal_loss, 'dice_loss': dice_loss, 'accs': accs}, epoch)
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