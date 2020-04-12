from sklearn.metrics import f1_score, accuracy_score
import torch
from pycocotools.cocoeval import COCOeval
import numpy as np
import torchvision
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torch.nn.functional as F

def metrics( net, dataloader, box_transform, epoch=1):
    net.eval()
    correct = 0
    total = 0
    average_precision = []
    images = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image, labels = data
            outputs = net(image.cuda()).cpu()
            boxes_tr = box_transform(labels)
            boxes_pr = box_transform(outputs.cpu().detach())
            metched =[False for x in boxes_tr]
            true_positives = 0
            for b in boxes_pr:
                for i, true_b in enumerate(boxes_tr):  
                    # if b['category_id'] != true_b['category_id']:
                    #     continue
                    if IoU(b['bbox'], true_b['bbox'])>0.1:
                        metched[i]=True
                        true_positives+=1
            average_precision.append(true_positives/(len(boxes_pr)+0.1**9))
            
            if i>len(dataloader)-5:
                object_range = 5*len(net.ratios)+net.classes
                outputs[:, ::object_range] = torch.sigmoid(outputs[:, ::object_range])
                boxes_pr = box_transform(outputs.cpu().detach())
                boxes_tr = box_transform(labels.cpu().detach())
                pilImage = torchvision.transforms.ToPILImage()(image[0,...])
                draw = ImageDraw.Draw(pilImage)
                for l in boxes_pr:
                    bbox = l['bbox']
                    draw.rectangle(((bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])), outline = 'red')
                    draw.text((bbox[0], bbox[1]-10), dataloader.cats[l['category_id']][1], outline = 'red')
                for l in boxes_tr:
                    bbox = l['bbox']
                    draw.rectangle(((bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])), outline = 'blue')
                    draw.text((bbox[0], bbox[1]-10), dataloader.cats[l['category_id']][1], outline = 'blue')
                images.append(np.array(pilImage))
                
    return sum(average_precision)/len(average_precision), images



def IoU(bboxDT, bboxGT):
    xA = max(bboxDT[0], bboxGT[0])
    yA = max(bboxDT[1], bboxGT[1])
    xB = min(bboxDT[0] + bboxDT[2], bboxGT[0] + bboxGT[2])
    yB = min(bboxDT[1] + bboxDT[3], bboxGT[1] + bboxGT[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = bboxDT[2] * bboxDT[2]
    boxBArea = bboxGT[2] * bboxGT[2]
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou