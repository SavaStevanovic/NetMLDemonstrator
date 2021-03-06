from PIL import ImageDraw
import numpy as np

def apply_detections(box_transform, outputs, labels, image, cats, thresh = 0.5, apply_nms = False):
    boxes_pr = box_transform(outputs, thresh)
    if apply_nms:
        boxes_pr = non_max_suppression(boxes_pr)
    boxes_tr = box_transform(labels)
    draw = ImageDraw.Draw(image)
    for l in boxes_pr:
        draw_box(draw, l['bbox'], cats[l['category_id']][1], 'red', 3)
    for l in boxes_tr:
        draw_box(draw, l['bbox'], cats[l['category_id']][1], 'blue', 1)
    return np.array(image)

def non_max_suppression(boxes):
    boxes = sorted(boxes, key = lambda x: -x['confidence'])
    filtered_boxes = []
    for b in boxes:
        suppress = len([0 for fb in filtered_boxes if fb['category_id']==b['category_id'] and IoU(fb['bbox'], b['bbox'])>0.5])
        if suppress==0:
            filtered_boxes.append(b)
    return filtered_boxes

def IoU(bboxDT, bboxGT):
    xA = max(bboxDT[0], bboxGT[0])
    yA = max(bboxDT[1], bboxGT[1])
    xB = min(bboxDT[0] + bboxDT[2], bboxGT[0] + bboxGT[2])
    yB = min(bboxDT[1] + bboxDT[3], bboxGT[1] + bboxGT[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = bboxDT[2] * bboxDT[3]
    boxBArea = bboxGT[2] * bboxGT[3]
    iou = 0
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea>0:
        iou = interArea / unionArea 
    return iou

def draw_box(draw, bbox, label, color, size):
    draw.rectangle(((bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])), outline = color, width=size)
    text_position = bbox[1]
    if text_position-10>0:
        text_position-=10
    draw.text((bbox[0]+3, text_position), label)
