import numpy as np

def join_images(images):
    if len(images)==0:
        return images
    hight = max(img.shape[0] for img in images)
    width = sum(img.shape[1] for img in images)
    joined_images = np.zeros((hight, width, 3), dtype=np.uint8)
    pos = 0
    for img in images:
        joined_images[:img.shape[0], pos:pos+img.shape[1]] = img[...,:3]
        pos += img.shape[1]
    return joined_images

def visualize_label(color_dict, img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(len(color_dict)):
        img_out[img == i,:] = color_dict[i]
    return img_out