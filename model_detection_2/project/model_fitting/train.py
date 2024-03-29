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
from torchvision.ops import nms, box_convert

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
        if train:
            optimizer.zero_grad()
        outputs = net(image.cuda())
        criterions = [criterion(outputs[i], labels[i].cuda()) for i in range(len(outputs))]
        loss, objectness_loss, size_loss, offset_loss, class_loss = (sum(x) for x in zip(*criterions))
        if train:
            loss.backward()
            optimizer.step()
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
    map = 0
    if not train:
        mapc = map_metrics.compute()
        pprint.pprint(mapc)
        map = mapc["map"].item()
        
    return losses/data_len, total_objectness_loss/data_len, total_size_loss/data_len, total_offset_loss/data_len, total_class_loss/data_len, images, map

def new_func(net, box_transform, map_metrics, labels, outputs):
    box_labels = box_transform([labels[0].cpu()[0].numpy()])
    box_outs = [out[0].cpu().detach().unsqueeze(0).numpy() for out in outputs]
    boxes_pr = []
    for l, out in enumerate(box_outs):
        boxes_pr += box_transform(out, threshold = 0.20, depth = l)
    preds = box_to_torch(boxes_pr, net.classes)
    # boxes = [[p[0], p[1], p[0] + p[2], p[1]+ p[3]] for p in preds["boxes"]]
    score_ids = []
    if len(preds["boxes"]):
        score_ids = nms(box_convert(preds["boxes"], "xywh", "xyxy"), scores = preds["scores"].to(torch.float64), iou_threshold = 0.5)
    labs = box_to_torch(box_labels, net.classes)
    map_metrics.update([{k: v[score_ids] for k, v in preds.items()}], [labs])

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
        loss, objectness_loss, size_loss, offset_loss, class_loss, samples, _ = fit_epoch(net, trainloader, train_config.learning_rate, True, epoch=epoch)
        writer.add_scalars('Train/Metrics', {'objectness_loss': objectness_loss, 'size_loss':size_loss, 'offset_loss':offset_loss, 'class_loss':class_loss}, epoch)
        writer.add_scalar('Train/Metrics/loss', loss, epoch)
        grid = join_images(samples)
        writer.add_images('train_sample', grid, epoch, dataformats='HWC')
        
        loss, objectness_loss, size_loss, offset_loss, class_loss, samples, map = fit_epoch(net, validationloader, train_config.learning_rate, False, epoch)
        writer.add_scalars('Validation/Metrics', {'objectness_loss': objectness_loss, 'size_loss':size_loss, 'offset_loss':offset_loss, 'class_loss':class_loss}, epoch)
        writer.add_scalar('Validation/Metrics/loss', loss, epoch)
        writer.add_scalar('Validation/Metrics/MAP', map, epoch)
        grid = join_images(samples)
        writer.add_images('validation_sample', grid, epoch, dataformats='HWC')

        os.makedirs((chp_dir), exist_ok=True)
        if train_config.best_metric < map:
            train_config.iteration_age = 0
            train_config.best_metric = map
            print('Epoch {}. Saving model with metric: {}'.format(epoch, map))
            torch.save(net, checkpoint_name_path.replace('.pth', '_final.pth'))
        else:
            train_config.iteration_age += 1
            print('Epoch {} metric: {}'.format(epoch, map))
            net.unlock_layer()
            net.unlock_layer()
        if (train_config.iteration_age-2) and ((train_config.iteration_age-2) % lower_learning_period) == 0:
            net.grad_backbone(True)
            print("Model unfrozen")
        if train_config.iteration_age and (train_config.iteration_age % lower_learning_period) == 0:
            train_config.learning_rate *= 0.5
            print("Learning rate lowered to {}".format(
                train_config.learning_rate))
        
        train_config.epoch = epoch+1
        train_config.save(checkpoint_conf_path)
        torch.save(net, checkpoint_name_path)
        torch.save(net.state_dict(), checkpoint_name_path.replace(
            '.pth', '_final_state_dict.pth'))
    net = torch.load(
        "checkpoints/YoloV2/64/0,5-1,0-2,0/Coco_checkpoints_final.pth")
    print('Finished Training')
    loss, objectness_loss, size_loss, offset_loss, class_loss, samples, map = fit_epoch(
        net, testloader, train_config.learning_rate, False, train_config.epoch)
    writer.add_scalars('Test/Metrics', {'objectness_loss': objectness_loss, 'size_loss': size_loss,
                                        'offset_loss': offset_loss, 'class_loss': class_loss}, train_config.epoch)
    writer.add_scalar('Test/Metrics/loss', loss, train_config.epoch)
    writer.add_scalar('Test/Metrics/MAP', map, train_config.epoch)
    grid = join_images(samples)
    writer.add_images('Test_sample', grid,
                      train_config.epoch, dataformats='HWC')
    return map
