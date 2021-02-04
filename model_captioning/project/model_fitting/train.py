import torch
import os
from model_fitting.losses import SegmentationLoss
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
import seaborn as sn

def get_acc(output, label):
    label_trim = label[label!=0]
    acc = (label_trim == output[:len(label_trim)]).float().sum() / len(label_trim)
    return acc.item()

def get_output_text(vectorizer, output):
    eos_index = vectorizer.vocab.index(vectorizer.eos_token)
    if eos_index in output:
        output = output[:output.index(eos_index)]
    output_text = ' '.join([vectorizer.vocab[x] for x in output])
    return output_text


def output_to_image(lab):
    lab = (lab>0.5).int().squeeze(0)
    color_map = [[0, 0, 0], [0, 0.99, 0]]
    lab = visualize_label(color_map, lab.numpy())
    
    return lab

def fit_epoch(net, dataloader, lr_rate, train, epoch=1):
    if train:
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    else:
        net.eval()
    cm = metrics.RunningConfusionMatrix()
    criterion = torch.nn.CrossEntropyLoss()
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    losses = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    output_images = []
    label_images = []
    outputs_vector = []
    label_vector = []
    accs = []
    f1_score = 0
    skip_count = 0

    for i, data in enumerate(tqdm(dataloader)):
        image, labels = data
        if image is None:
            continue    
        # print(image.shape)
        if train:
            optimizer.zero_grad()
        labels_cuda = labels.cuda()

        state = None
        d = image.cuda()
        outputs = []
        for j in range(labels_cuda.shape[1]):
            output, state = net(d, state)
            outputs.append(output.unsqueeze(-1))
            d = labels_cuda[:, j]        
        outputs = torch.cat(outputs[1:], -1)

        loss = criterion(outputs, labels_cuda[:, 1:])
        if train:
            loss.backward()
            optimizer.step()
        losses += loss.item()

        for j in range(outputs.shape[0]):
            acc = get_acc(outputs[j].argmax(0), labels_cuda[j, 1:])
            accs.append(acc)
        
        if i>=len(dataloader)-10:
            image = image[0]
            for t, m, s in zip(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
                t.mul_(s).add_(m)
            image = image.permute(1,2,0).detach().cpu().numpy()
            lab_text = get_output_text(net.vectorizer, labels[0, 1:].detach().cpu().numpy().tolist())
            out_text = get_output_text(net.vectorizer, outputs[0].detach().argmax(0).cpu().numpy().tolist())

            plt.gcf().subplots_adjust(bottom=0.15)
            plt.imshow(image)
            plt.xlabel(out_text + '\n' + lab_text)
            output_images.append(plt_to_np(plt))

    return losses/len(accs), output_images, sum(accs)/len(accs)

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
    summary(net, (3, *net.input_size))
    writer = SummaryWriter(os.path.join('logs', model_dir_header))
    images, _ = next(iter(trainloader))

    with torch.no_grad():
        writer.add_graph(net, images[:2].cuda())
    for epoch in range(train_config.epoch, epochs):
        loss, output_images, accs = fit_epoch(net, trainloader, train_config.learning_rate, train = True, epoch=epoch)
        writer.add_scalars('Train/Metrics', {'loss': loss, 'accs': accs}, epoch)
        grid = join_images(output_images)
        writer.add_images('train_outputs', grid, epoch, dataformats='HWC')

        loss, output_images, accs = fit_epoch(net, validationloader, None, train = False, epoch=epoch)
        writer.add_scalars('Validation/Metrics', {'loss': loss, 'accs': accs}, epoch)
        grid = join_images(output_images)
        writer.add_images('validation_outputs', grid, epoch, dataformats='HWC')

        os.makedirs((chp_dir), exist_ok=True)
        if train_config.best_metric < accs:
            train_config.iteration_age = 0
            train_config.best_metric = accs
            print('Epoch {}. Saving model with metric: {}'.format(epoch, accs))
            torch.save(net, checkpoint_name_path.replace('.pth', '_final.pth'))
        else:
            train_config.iteration_age+=1
            print('Epoch {} metric: {}'.format(epoch, accs))
        if train_config.iteration_age==lower_learning_period:
            train_config.learning_rate*=0.5
            train_config.iteration_age=0
            print("Learning rate lowered to {}".format(train_config.learning_rate))
        train_config.epoch = epoch+1
        train_config.save(checkpoint_conf_path)
        torch.save(net, checkpoint_name_path)
        torch.save(net.state_dict(), checkpoint_name_path.replace('.pth', '_final_state_dict.pth'))
    print('Finished Training')