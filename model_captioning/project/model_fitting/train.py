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
from nlgeval import NLGEval

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
    criterion = torch.nn.CrossEntropyLoss(ignore_index = net.vectorizer.vocab.index(net.vectorizer.pad_token))
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    losses = 0.0
    output_images = []
    accs = []
    label_texts = []
    output_texts = []
    nlgeval = NLGEval()
    out_file = open("output.txt", "w")
    lab_file = open("labels.txt", "w")

    for i, data in enumerate(tqdm(dataloader)):
        image, labels, label_lens = data
        # labels_lens (seq_length, batch_size)
        if image is None:
            continue    
        if train:
            optimizer.zero_grad()
        labels_cuda = labels.cuda()

        state = None
        outputs = torch.zeros((labels.shape[0], len(net.vectorizer.vocab), labels.shape[1]), device = 'cuda')
        d = image.cuda()
        start = 0
        init_axis = 0
        for l in label_lens:
            end = l[0]
            for j in range(start, end):
                output, state = net(d, state)
                outputs[-len(output):, :, j] = output
                d = labels_cuda[init_axis:, j] 
            state = (state[0][l[1]:], state[1][l[1]:])
            init_axis += l[1]
            d = labels_cuda[init_axis:, j]
            start = end

        loss = criterion(outputs[:, :, 1:], labels_cuda[:, 1:])
        if train:
            loss.backward()
            optimizer.step()
        losses += loss.item()*labels.shape[0]

        for j in range(outputs.shape[0]):
            acc = get_acc(outputs[j, :, 1:].argmax(0), labels_cuda[j, 1:])
            accs.append(acc)
            lab_text = get_output_text(net.vectorizer, labels[j, 1:].detach().cpu().numpy().tolist())
            out_text = get_output_text(net.vectorizer, outputs[j, :, 1:].detach().argmax(0).cpu().numpy().tolist())
            lab_file.write(lab_text + os.linesep)
            out_file.write(out_text + os.linesep)

        if i>=len(dataloader)-10:
            image = image[0]
            for t, m, s in zip(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
                t.mul_(s).add_(m)
            image = image.permute(1,2,0).detach().cpu().numpy()
            

            plt.gcf().subplots_adjust(bottom=0.15)
            plt.imshow(image)
            plt.xlabel(out_text + '\n' + lab_text)
            output_images.append(plt_to_np(plt))

    out_file.close()
    lab_file.close()
    metrics_dict = nlgeval.compute_individual_metrics('labels.txt', 'output.txt')
    return losses/len(accs), output_images, sum(accs)/len(accs), metrics_dict

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
    images, _, _ = next(iter(trainloader))

    with torch.no_grad():
        writer.add_graph(net, images[:2].cuda())
    for epoch in range(train_config.epoch, epochs):
        loss, output_images, accs, metrics_dict = fit_epoch(net, trainloader, train_config.learning_rate, train = True, epoch=epoch)
        writer.add_scalars('Train/Metrics', metrics_dict, epoch)
        writer.add_scalar('Train/Metrics_loss', loss, epoch)
        writer.add_scalar('Train/Metrics_accs', accs, epoch)
        grid = join_images(output_images)
        writer.add_images('train_outputs', grid, epoch, dataformats='HWC')

        loss, output_images, accs, metrics_dict = fit_epoch(net, validationloader, None, train = False, epoch=epoch)
        writer.add_scalars('Validation/Metrics', metrics_dict, epoch)
        writer.add_scalar('Validation/Metrics_loss', loss, epoch)
        writer.add_scalar('Validation/Metrics_accs', accs, epoch)
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
        if train_config.iteration_age and (train_config.iteration_age % lower_learning_period) == 0:
            train_config.learning_rate*=0.5
            print("Learning rate lowered to {}".format(train_config.learning_rate))
        if train_config.iteration_age == 2 * lower_learning_period:
            net.grad_backbone(True)
            print("Model unfrozen")
        train_config.epoch = epoch+1
        train_config.save(checkpoint_conf_path)
        torch.save(net, checkpoint_name_path)
        torch.save(net.state_dict(), checkpoint_name_path.replace('.pth', '_final_state_dict.pth'))
    print('Finished Training')