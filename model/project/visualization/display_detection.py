from PIL import ImageDraw
import torchvision
import numpy as np

def apply_detections(box_transform, outputs, labels, image, cats):
    boxes_pr = box_transform(outputs)
    boxes_tr = box_transform(labels)
    pil_image = torchvision.transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(pil_image)
    for l in boxes_pr:
        draw_box(draw, l['bbox'], cats[l['category_id']][1], 'red')
    for l in boxes_tr:
        draw_box(draw, l['bbox'], cats[l['category_id']][1], 'blue')
    return np.array(pil_image)
    
def draw_box(draw, bbox, label, color):
    draw.rectangle(((bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])), outline = color)
    draw.text((bbox[0], bbox[1]-10), label, outline = color)

