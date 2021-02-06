import requests
import shutil 
from PIL import Image  
import os
from tqdm import tqdm
import torchvision.transforms as transforms
import multiprocessing

def download_image(filename, image_url):
        try:
            im = Image.open(requests.get(image_url, stream=True, timeout=2.5).raw)
            im = transforms.functional.resize(im, 256, Image.ANTIALIAS)
            im.save(filename)
            print("Downloaded {}".format(filename))
        except Exception as e:
            print(e)

urls_file = open('SBU_captioned_photo_dataset_urls.txt', 'r')
urls = urls_file.readlines()
os.makedirs('images', exist_ok=True)
args = [('images/{}.png'.format(i), url) for i, url in enumerate(urls)]

with multiprocessing.Pool(processes=24) as pool:
    pool.starmap(download_image, args)