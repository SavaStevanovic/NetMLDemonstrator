import torch
from visualization.display_detection import apply_detections
from pycocotools.cocoeval import COCOeval

def metrics( net, dataloader, box_transform, epoch=1):
    net.eval()
    correct = 0
    total = 0
    true_boxes_count = 0
    images = []
    det_boxes = []
    ref_boxes = [[] for _ in range(len(dataloader))]
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image, labels = data
            outputs = net(image.cuda()).cpu()

            boxes_pr = box_transform(outputs.cpu().detach(), 0.5)
            boxes_tr = box_transform(labels.cpu().detach())
            true_boxes_count+=len(boxes_tr)
            for x in boxes_pr:
                x['image'] = i
            for x in boxes_tr:
                x['seen'] = 0
            det_boxes+=boxes_pr
            ref_boxes[i]=boxes_tr
            
            if i>=len(dataloader)-5:
                pilImage = apply_detections(box_transform, outputs.cpu().detach(), labels.cpu().detach(), image[0,...], dataloader.cats)
                images.append(pilImage)
    predicted_boxes_count = len(det_boxes)
    det_boxes = sorted(det_boxes, key = lambda x: -x['confidence'])
    true_prediction = [0 for _ in range(len(det_boxes))]
    predicted = [0 for _ in range(len(det_boxes))]
    for i, x in enumerate(det_boxes):
        for l in ref_boxes[x['image']]:
            if x['category_id'] == l['category_id'] and IoU(x['bbox'], l['bbox'])>0.5:
                if l['seen']==0:
                    l['seen']=1
                    predicted[i]+=1
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

    recall = [0 for _ in range(predicted_boxes_count)]
    for i, p in enumerate(predicted):
        s+=p
        recall[i]=s/true_boxes_count

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
    return metric, images



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