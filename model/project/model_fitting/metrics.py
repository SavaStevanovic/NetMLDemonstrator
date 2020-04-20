import torch
from visualization.display_detection import apply_detections
from pycocotools.cocoeval import COCOeval

def metrics( net, dataloader, box_transform, epoch=1):
    net.eval()
    correct = 0
    total = 0
    metric = []
    images = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image, labels = data
            outputs = net(image.cuda()).cpu()

            boxes_pr = box_transform(outputs.cpu().detach())
            boxes_tr = box_transform(labels.cpu().detach())
            metched =[False for x in boxes_tr]
            boxes_pr = sorted(boxes_pr, key= lambda x: -x['confidence'])
            true_positives = 0
            for b in boxes_pr:
                box_matched = False
                for j, true_b in enumerate(boxes_tr):  
                    if b['category_id'] == true_b['category_id'] and IoU(b['bbox'], true_b['bbox'])>0.5:
                        box_matched=True
                        metched[j]=True
                        break
                true_positives+=box_matched
            recall = sum(metched)/(len(boxes_tr)+0.1**9)
            precision = true_positives/(len(boxes_pr)+0.1**9)
            if precision+recall==0:
                metric.append(1.0)
            else:
                f1 = 2*recall*precision/(recall+precision)
                metric.append(f1)
            
            if i>len(dataloader)-5:
                pilImage = apply_detections(box_transform, outputs.cpu().detach(), labels.cpu().detach(), image[0,...], dataloader.cats)
                images.append(pilImage)
                
    return sum(metric)/len(metric), images



def IoU(bboxDT, bboxGT):
    xA = max(bboxDT[0], bboxGT[0])
    yA = max(bboxDT[1], bboxGT[1])
    xB = min(bboxDT[0] + bboxDT[2], bboxGT[0] + bboxGT[2])
    yB = min(bboxDT[1] + bboxDT[3], bboxGT[1] + bboxGT[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = bboxDT[2] * bboxDT[2]
    boxBArea = bboxGT[2] * bboxGT[2]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou