import torch
from visualization.display_detection import apply_detections
from pycocotools.cocoeval import COCOeval
from model_fitting.losses import YoloLoss

def metrics(net, dataloader, box_transform, epoch=1):
    net.eval()
    criterion = YoloLoss(ranges = net.ranges)
    losses = 0.0
    total_objectness_loss = 0.0
    total_size_loss = 0.0
    total_offset_loss = 0.0
    total_class_loss = 0.0
    images = []
    det_boxes = []
    ref_boxes = [[] for _ in range(len(dataloader))]
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image, labels = data
            outputs_cuda = net(image.cuda())
            labels_cuda = labels.cuda()
            loss, objectness_loss, size_loss, offset_loss, class_loss = criterion(outputs_cuda, labels_cuda)
            total_objectness_loss += objectness_loss
            total_size_loss += size_loss
            losses += loss.item()
            total_offset_loss += offset_loss
            total_class_loss += class_loss
            outputs = outputs_cuda.detach().cpu()

            boxes_pr = box_transform(outputs.squeeze(0), 0.5)
            boxes_tr = box_transform(labels.squeeze(0))
            for x in boxes_pr:
                x['image'] = i
            for x in boxes_tr:
                x['seen'] = 0
            det_boxes+=boxes_pr
            ref_boxes[i]=boxes_tr
            
            if i>=len(dataloader)-5:
                pilImage = apply_detections(box_transform, outputs[0], labels[0], image[0], dataloader.cats)
                images.append(pilImage)
    metric, _ = calculateMAP(det_boxes, ref_boxes, net.classes)
    data_len = len(dataloader)
    
    return metric, losses/data_len, total_objectness_loss/data_len, total_size_loss/data_len, total_offset_loss/data_len, total_class_loss/data_len, images


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
            if l['seen']==0 and IoU(x['bbox'], l['bbox'])>0.5:
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

def voc_ap(rec, prec):
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
   
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


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