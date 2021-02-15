import torch
import os
from model_fitting.losses import SegmentationLoss
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from visualization.images_display import join_images, visualize_label
from model_fitting.configuration import TrainingConfiguration
import json
from functools import reduce
from torchsummary import summary
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
from utils import plt_to_np
from model_fitting import metrics
from operator import itemgetter
import seaborn as sn
from nlgeval import compute_metrics
from nltk.translate.bleu_score import corpus_bleu

def get_acc(output, label):
    label_trim = label[label!=0]
    acc = (label_trim == output[:len(label_trim)]).float().sum() / len(label_trim)
    return acc.item()

def get_output_text(vectorizer, output):
    eos_index = np.where(vectorizer.vocab == vectorizer.eos_token)[0][0]
    if eos_index in output:
        output = output[:np.where(output == eos_index)[0][0]]
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
    criterion = torch.nn.CrossEntropyLoss(ignore_index = np.where(net.vectorizer.vocab == net.vectorizer.pad_token)[0][0])
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    losses = 0.0
    att_losses = 0.0
    crit_losses = 0.0
    output_images = []
    accs = 0
    sample_count = 0
    label_texts = []
    output_texts = []
    references = []
    hypotheses = []
    out_file = open("output.txt", "w")
    lab_file_names = ["labels{}.txt".format(i) for i in range(5)]
    lab_files = [open(f, "w") for f in lab_file_names]

    for i, data in enumerate(tqdm(dataloader)):
        image, labels, all_labels = data
        # labels_lens (seq_length, batch_size)
        if image is None:
            continue    
        if train:
            optimizer.zero_grad()
        labels_cuda = labels.cuda()

        outputs, atts = net(image.cuda(), labels_cuda[:, :-1])

        att_loss = 10 * ((atts.mean((1,2)).unsqueeze(-1)-atts.sum(2)) ** 2).mean() 
        crit_loss = criterion(outputs, labels_cuda[:, 1:])
        loss = crit_loss + att_loss
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

        losses += loss.item() * labels.shape[0]
        att_losses += att_loss.item() * labels.shape[0]
        crit_losses += crit_loss.item() * labels.shape[0]

        sample_count += labels_cuda.shape[0]
        for j in range(outputs.shape[0]):
            accs += get_acc(outputs[j].argmax(0), labels_cuda[j, 1:])
            if not train:
                h = outputs[j].detach().argmax(0).cpu().numpy().tolist()
                hypotheses.append(h)
                lab_text = get_output_text(net.vectorizer, labels[j, 1:].detach().cpu().numpy().tolist())
                out_text = get_output_text(net.vectorizer, h)
                ls = []
                for p, l in enumerate(all_labels[j]):
                    ll = l[1:]
                    ls.append(ll)
                    l_text = get_output_text(net.vectorizer, ll)
                    lab_files[p].write(l_text + os.linesep)
                references.append(ls)
                out_file.write(out_text + os.linesep)

        if i>=len(dataloader)-10:
            image = image[0]
            lab_text = get_output_text(net.vectorizer, labels[0, 1:].detach().cpu().numpy().tolist())
            out_text = get_output_text(net.vectorizer, outputs[0].detach().argmax(0).cpu().numpy().tolist())
            for t, m, s in zip(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
                t.mul_(s).add_(m)
            image = image.permute(1,2,0).detach().cpu().numpy()
            

            plt.gcf().subplots_adjust(bottom=0.15)
            plt.imshow(image)
            plt.xlabel(out_text + '\n' + lab_text)
            output_images.append(plt_to_np(plt))

    out_file.close()
    for l in lab_files:
        l.close()
    metrics_dict = {}
    if not train:
        metrics_dict = compute_metrics(references=lab_file_names, hypothesis='output.txt', no_overlap=False, no_skipthoughts=True, no_glove=True)
        bleu4 = corpus_bleu(references, hypotheses)
    return losses/sample_count, att_losses/sample_count, crit_losses/sample_count, output_images, accs/sample_count, metrics_dict

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

    writer = SummaryWriter(os.path.join('logs', model_dir_header))

    images, label, _ = next(iter(trainloader))
    summary(net, [tuple(images.shape[1:]), tuple(label.shape[1:])])
    with torch.no_grad():
        writer.add_graph(net, (images[:2].cuda(), label[:2].cuda()))

    for epoch in range(train_config.epoch, epochs):
        loss, att_losses, crit_losses, output_images, accs, _ = fit_epoch(net, trainloader, train_config.learning_rate, train = True, epoch=epoch)
        writer.add_scalar('Train/Metrics_att_losses', att_losses, epoch)
        writer.add_scalar('Train/Metrics_crit_losses', crit_losses, epoch)
        writer.add_scalar('Train/Metrics_loss', loss, epoch)
        writer.add_scalar('Train/Metrics_accs', accs, epoch)
        grid = join_images(output_images)
        writer.add_images('train_outputs', grid, epoch, dataformats='HWC')

        loss, att_losses, crit_losses, output_images, accs, metrics_dict = fit_epoch(net, validationloader, None, train = False, epoch=epoch)
        writer.add_scalars('Validation/Metrics', metrics_dict, epoch)
        writer.add_scalar('Validation/Metrics_att_losses', att_losses, epoch)
        writer.add_scalar('Validation/Metrics_crit_losses', crit_losses, epoch)
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