import torch
from visualization import apply_output
from pycocotools.cocoeval import COCOeval
from model_fitting.losses import YoloLoss
from tqdm import tqdm
from functools import reduce
from torchvision.transforms.functional import to_pil_image
from visualization.output_transform import get_mapped_image 
import numpy as np

def metrics(net, dataloader, postprocessing, epoch=1):
    net.eval()
    criterion = torch.nn.MSELoss()
    losses = 0.0
    total_pafs_loss = 0.0
    total_maps_loss = 0.0
    output_images = []
    label_images = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            image, labels = data
            pafs_output, maps_output = net(image.cuda())
            paf_label = labels[0].cuda()
            map_label = labels[1].cuda()
            pafs_loss = torch.stack([(paf_label>0).int() * criterion(paf_label, o) for o in pafs_output], dim=0).sum()
            maps_loss = torch.stack([(map_label>0).int() * criterion(map_label, o) for o in maps_output], dim=0).sum()
            loss = pafs_loss + maps_loss
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

def calculateMAP(det_boxes, ref_boxes, classes):
    class_det_boxes = [[] for _ in classes]
    for x in det_boxes:
        class_det_boxes[x['category_id']].append(x)
    class_average_precision = [calculateAP(class_det_boxes[i], ref_boxes, i) for i in range(len(classes))]
    mean_average_precision = sum(class_average_precision)/len(class_average_precision)
    return mean_average_precision, class_average_precision

def calculateAP(det_boxes, ref_boxes, class_id):
    ref_boxes = [[x for x in image_boxes if x['category_id'] == class_id] for image_boxes in ref_boxes]

    predicted_boxes_count = len(det_boxes)
    det_boxes = sorted(det_boxes, key = lambda x: -x['confidence'])
    true_prediction = [0 for _ in range(predicted_boxes_count)]
    for i, x in enumerate(det_boxes):
        image_boxes = ref_boxes[x['image']]
        for l in image_boxes:
            if l['seen']==0 and apply_output.IoU(x['bbox'], l['bbox'])>0.5:
                l['seen']=1
                true_prediction[i]=1
                break
    precision = [0 for _ in range(predicted_boxes_count)]
    s=0
    for i, p in enumerate(true_prediction):
        s+=p
        precision[i]=s/(i+1)

    interpolated_precision = [0 for _ in range(predicted_boxes_count)]
    maxs=0
    for i, p in enumerate(precision[::-1]):
        maxs = max(maxs, p)
        interpolated_precision[-i-1]=maxs

    true_boxes_count = sum([len(x) for x in ref_boxes])
    recall = [0 for _ in range(predicted_boxes_count)]
    for i, p in enumerate(true_prediction):
        s+=p
        recall[i]=s/max(true_boxes_count, 1)

    period = 0.01
    start = 0    
    i=0
    indeces =[]
    while i<len(recall) and start<=1.0:
        if recall[i]>start:
            indeces.append(interpolated_precision[i])
            start+=period
        else:
            i+=1

    metric = sum(indeces)/(1/period+1)
    return metric

