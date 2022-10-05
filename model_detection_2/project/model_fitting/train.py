from collections import defaultdict
from itertools import chain
from operator import methodcaller
import pprint
import torch
import os
from model_fitting.losses import YoloLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from visualization.apply_output import apply_detections
from visualization.images_display import join_images
from model_fitting.configuration import TrainingConfiguration
from torchsummary import summary
from torchvision.transforms.functional import to_pil_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from visualization.output_transform import TargetTransformToBoxes


def fit_epoch(net, dataloader, lr_rate, train, epoch=1):
    if train:
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    else:
        net.eval()
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    box_transform=TargetTransformToBoxes(prior_box_sizes=net.prior_box_sizes, classes=net.classes, ratios=net.ratios, strides=net.strides)
    optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    criterion = YoloLoss(ranges = net.ranges)
    losses = 0.0
    total_objectness_loss = 0.0
    total_size_loss = 0.0
    total_offset_loss = 0.0
    total_class_loss = 0.0
    images = []
    map_metrics = MeanAveragePrecision(box_format="xywh", class_metrics=True)
    for i, data in enumerate(tqdm(dataloader)):
        (image, labels), data_labels = data
        data_labels = [dl.split("%") for dl in data_labels]
        data_labels_ids = [[net.ranges.classes[net.classes.index(l)] for l in ls] for ls in data_labels]
        # data_labels_neg_ids = [[x for x in net.ranges.classes if x not in dl] for dl in data_labels_ids]
        if train:
            optimizer.zero_grad()
        outputs = net(image.cuda())
        criterions = [
            criterion(
                outputs[i][:, :, list(range(5)) + data_labels_ids[i], ...], 
                labels[i][:, :, list(range(5)) + data_labels_ids[i], ...].cuda()
            ) for i in range(len(outputs))
        ]
        loss, objectness_loss, size_loss, offset_loss, class_loss = (sum(x) for x in zip(*criterions))
        if train:
            loss.backward()
            optimizer.step()
            
        # outputs = net(image.cuda())
        # outputs = [out.detach() for out in outputs]
        total_objectness_loss += objectness_loss
        total_size_loss += size_loss
        losses += loss.item()
        total_offset_loss += offset_loss
        total_class_loss += class_loss
        if not train:
            new_func(net, box_transform, map_metrics, labels, outputs)
        if i>=len(dataloader)-5:
            outs = [out[0].cpu().detach().unsqueeze(0).numpy() for out in outputs]
            labs = [labels[0].cpu()[0].numpy()]
            pilImage = apply_detections(box_transform, outs, labs, to_pil_image(image[0]))
            images.append(pilImage)
        
    data_len = len(dataloader)
    if not train:
        pprint.pprint(map_metrics.compute())
        
    return losses/data_len, total_objectness_loss/data_len, total_size_loss/data_len, total_offset_loss/data_len, total_class_loss/data_len, images

def new_func(net, box_transform, map_metrics, labels, outputs):
    box_labels = box_transform([labels[0].cpu()[0].numpy()])
    box_outs = [out[0].cpu().detach().unsqueeze(0).numpy() for out in outputs]
    boxes_pr = []
    for l, out in enumerate(box_outs):
        boxes_pr += box_transform(out, threshold = 0.20, depth = l)
    labs = box_to_torch(box_labels, net.classes)
    map_metrics.update([box_to_torch(boxes_pr, net.classes)], [labs])

def box_to_torch(boxes: list, classes: list):
    dd = dict(
        boxes=[],
        scores=[],
        labels=[]
    )
    boxes = [box_form(b, classes)for b in boxes]
    dict_items = map(methodcaller('items'), boxes)
    for k, v in chain.from_iterable(dict_items):
        dd[k].append(v)
    dd = {k: torch.tensor(v) for k, v in dd.items()}
    
    return dd

def box_form(box: dict, classes: list):
    out = dict(
        boxes=box["bbox"],
        scores=box["confidence"],
        labels=classes.index(box["category"])
    )
    
    return out

def fit(net, trainloader, validationloader, dataset_name, epochs=1000, lower_learning_period=10):
    model_dir_header = net.get_identifier()
    chp_dir = os.path.join('checkpoints', model_dir_header)
    checkpoint_name_path = os.path.join(chp_dir, '{}_checkpoints.pth'.format(dataset_name))
    checkpoint_conf_path = os.path.join(chp_dir, '{}_configuration.json'.format(dataset_name))
    train_config = TrainingConfiguration()
    if os.path.exists(chp_dir):
        net = torch.load(checkpoint_name_path)
        train_config.load(checkpoint_conf_path)
    net.cuda()
    # summary(net, (3, 224, 224))
    writer = SummaryWriter(os.path.join('logs', model_dir_header))
    for epoch in range(train_config.epoch, epochs):
        loss, objectness_loss, size_loss, offset_loss, class_loss, samples = fit_epoch(net, trainloader, train_config.learning_rate, True, epoch=epoch)
        writer.add_scalars('Train/Metrics', {'objectness_loss': objectness_loss, 'size_loss':size_loss, 'offset_loss':offset_loss, 'class_loss':class_loss}, epoch)
        writer.add_scalar('Train/Metrics/loss', loss, epoch)
        grid = join_images(samples)
        writer.add_images('train_sample', grid, epoch, dataformats='HWC')
        
        loss, objectness_loss, size_loss, offset_loss, class_loss, samples = fit_epoch(net, validationloader, train_config.learning_rate, False, epoch)
        writer.add_scalars('Validation/Metrics', {'objectness_loss': objectness_loss, 'size_loss':size_loss, 'offset_loss':offset_loss, 'class_loss':class_loss}, epoch)
        writer.add_scalar('Validation/Metrics/loss', loss, epoch)
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
        if train_config.iteration_age == lower_learning_period-1:
            net.unlock_layer()
            print("Model unfrozen")
        if train_config.iteration_age and (train_config.iteration_age % lower_learning_period) == 0:
            train_config.learning_rate*=0.5
            print("Learning rate lowered to {}".format(train_config.learning_rate))
        
        train_config.epoch = epoch+1
        train_config.save(checkpoint_conf_path)
        torch.save(net, checkpoint_name_path)
        torch.save(net.state_dict(), checkpoint_name_path.replace('.pth', '_final_state_dict.pth'))
    print('Finished Training')
    return best_map