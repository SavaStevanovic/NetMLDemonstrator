import torch
import os
from model_fitting.losses import SegmentationLoss
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

def fit_epoch(net, dataloader, lr_rate, postprocessing, train, epoch=1):
    if train:
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    else:
        net.eval()
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    criterion = torch.nn.MSELoss(reduction='none')
    losses = 0.0
    total_pafs_loss = 0.0
    total_maps_loss = 0.0
    output_images = []
    label_images = []
    map_outputs_images = []
    map_labels_images = []
    paf_outputs_images = []
    paf_labels_images = []


    for i, data in enumerate(tqdm(dataloader)):
        image, pafs, maps, mask = data
        # image = (image+0.5)
        # labels_cuda = labels.cuda()
        if train:
            optimizer.zero_grad()
        pafs_output, maps_output = net(image.cuda())
        paf_label = pafs.cuda()
        map_label = maps.cuda()
        mask_cuda = mask.cuda()
        # pafs_loss = torch.stack([mask_cuda * criterion(o, paf_label) for o in pafs_output], dim=0).mean()
        # maps_loss = torch.stack([mask_cuda * criterion(o, map_label) for o in maps_output], dim=0).mean()
        # loss = maps_loss + pafs_loss
        loss, pafs_loss, maps_loss = compute_loss(pafs_output, maps_output, paf_label, map_label, mask_cuda)
        if train:
            loss.backward()
            optimizer.step()
        total_pafs_loss += pafs_loss.sum()
        total_maps_loss += maps_loss.sum()
        losses += loss.item()

        if i>=len(dataloader)-5:
            mask = mask.detach()
            image = image.detach()
            with torch.no_grad():
                pafs_output = [F.interpolate(x, image.shape[2:], mode='bilinear', align_corners=True) for x in pafs_output]
                maps_output = [F.interpolate(x, image.shape[2:], mode='bilinear', align_corners=True) for x in maps_output]

            mapped_image = get_mapped_image((1 - mask) * image, pafs, maps, postprocessing, dataloader.skeleton, dataloader.parts)
            label_images.append(np.array(mapped_image))

            plt.imshow(maps_output[-1].clamp(0, 1).cpu().numpy()[0, -1])
            plt.imshow(image[0].permute(1,2,0), alpha=0.2)
            map_outputs_images.append(plt_to_np(plt))


            plt.imshow(maps.cpu().clamp(0, 1).numpy()[0, -1])
            plt.imshow(image[0].permute(1,2,0), alpha=0.2)
            map_labels_images.append(plt_to_np(plt))
            
            plt.imshow(((pafs_output[-1].clamp(-1, 1).cpu().numpy()[0, :]+1)/2).mean(0))
            plt.imshow(image[0].permute(1,2,0), alpha=0.2)
            paf_outputs_images.append(plt_to_np(plt))


            plt.imshow(((pafs.clamp(-1, 1).cpu().numpy()[0, :]+1)/2).mean(0))
            plt.imshow(image[0].permute(1,2,0), alpha=0.2)
            paf_labels_images.append(plt_to_np(plt))

            mapped_image = get_mapped_image(image, pafs_output[-1], maps_output[-1], postprocessing, dataloader.skeleton, dataloader.parts)
            output_images.append(np.array(mapped_image))

    data_len = len(dataloader)
    return losses/data_len, total_pafs_loss/data_len, total_maps_loss/data_len, output_images, label_images, map_outputs_images, map_labels_images, paf_outputs_images, paf_labels_images

def mean_square_error(pred, target):
    assert pred.shape == target.shape, 'x and y should in same shape'
    return torch.sum((pred - target) ** 2) / target.nelement()

def compute_loss(pafs_ys, heatmaps_ys, pafs_t, heatmaps_t, ignore_mask):
    total_loss = 0
    # compute loss on each stage
    heatmap_loss_log, loss = pose_loss(heatmaps_ys, heatmaps_t, ignore_mask)
    total_loss             += loss
    paf_loss_log    , loss = pose_loss(pafs_ys    , pafs_t    , ignore_mask    )
    total_loss             += loss

    return total_loss, np.array(paf_loss_log), np.array(heatmap_loss_log)

def pose_loss(ys, t, masks):
    masks = masks.repeat([1, t.shape[1], 1, 1])
    sum_loss = 0
    loss_log = []
    stage_t = t.clone()
    stage_masks = masks.clone()
    for y in ys:
        if stage_t.shape != y.shape:
            y = F.interpolate(y, stage_t.shape[2:], mode='bilinear', align_corners=True)
    
        y[stage_masks > 0.5] = stage_t.detach()[stage_masks > 0.5]        

        loss = mean_square_error(y, stage_t)
        sum_loss += loss
        loss_log.append(loss.item())
    return loss_log, sum_loss

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
    summary(net, (3, 416, 416))
    writer = SummaryWriter(os.path.join('logs', model_dir_header))
    # images, _, _, _ = next(iter(trainloader))

    # with torch.no_grad():
    #     writer.add_graph(net, images[:2].cuda())
    for epoch in range(train_config.epoch, epochs):
        loss, total_pafs_loss, total_maps_loss, output_images, label_images, map_outputs_images, map_labels_images, paf_outputs_images, paf_labels_images = fit_epoch(net, trainloader, train_config.learning_rate, postprocessing, train = True, epoch=epoch)
        writer.add_scalars('Train/Metrics', {'total_pafs_loss': total_pafs_loss, 'total_maps_loss': total_maps_loss}, epoch)
        writer.add_scalar('Train/Metrics/loss', loss, epoch)
        grid = join_images(label_images)
        writer.add_images('train_labels', grid, epoch, dataformats='HWC')
        grid = join_images(output_images)
        writer.add_images('train_outputs', grid, epoch, dataformats='HWC')
        grid = join_images(map_outputs_images)
        writer.add_images('train_map_outputs_images', grid, epoch, dataformats='HWC')
        grid = join_images(map_labels_images)
        writer.add_images('train_map_labels_images', grid, epoch, dataformats='HWC')
        grid = join_images(paf_outputs_images)
        writer.add_images('train_paf_outputs_images', grid, epoch, dataformats='HWC')
        grid = join_images(paf_labels_images)
        writer.add_images('train_paf_labels_images', grid, epoch, dataformats='HWC')

        loss, total_pafs_loss, total_maps_loss, output_images, label_images, map_outputs_images, map_labels_images, paf_outputs_images, paf_labels_images = fit_epoch(net, validationloader, None, postprocessing, train = False, epoch=epoch)
        writer.add_scalars('Validation/Metrics', {'total_pafs_loss': total_pafs_loss, 'total_maps_loss': total_maps_loss}, epoch)
        writer.add_scalar('Validation/Metrics/loss', loss, epoch)
        grid = join_images(label_images)
        writer.add_images('validation_labels', grid, epoch, dataformats='HWC')
        grid = join_images(output_images)
        writer.add_images('validation_outputs', grid, epoch, dataformats='HWC')
        grid = join_images(map_outputs_images)
        writer.add_images('validation_map_outputs_images', grid, epoch, dataformats='HWC')
        grid = join_images(map_labels_images)
        writer.add_images('validation_map_labels_images', grid, epoch, dataformats='HWC')
        grid = join_images(paf_outputs_images)
        writer.add_images('validation_paf_outputs_images', grid, epoch, dataformats='HWC')
        grid = join_images(paf_labels_images)
        writer.add_images('validation_paf_labels_images', grid, epoch, dataformats='HWC')

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