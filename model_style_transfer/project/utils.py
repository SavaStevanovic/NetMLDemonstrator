import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def plt_to_np(plt_img):
    buffer_ = BytesIO()
    plt_img.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = Image.open(buffer_)
    # image.save('test.png')
    plt_img.close()
    return np.array(image)