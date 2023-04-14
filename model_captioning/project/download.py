import os
import urllib.request
from urllib.parse import urlparse
import multiprocessing
from tqdm import tqdm

# Define the file path containing the image URLs
url_file_path = "/Data1/Data/captioning/SBU/SBUCaptionedPhotoDataset/dataset/SBU_captioned_photo_dataset_urls.txt"

# Define the folder path to save the downloaded images
folder_path = "/Data1/Data/captioning/SBU/dataset"

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Open the URL file and read its contents
with open(url_file_path, "r") as f:
    url_list = f.readlines()

# Define a function to download an image from a URL


def download_image(url):
    try:
        # Get the last part of the URL as the file name
        url_parts = urlparse(url)
        file_name = os.path.join(folder_path, os.path.basename(url_parts.path))

        # Download the image from the URL
        urllib.request.urlretrieve(url.strip(), file_name)

        return file_name
    except Exception as e:
        # print(f"Error downloading {url}: {e}")
        return None


# Define the number of processes to use
num_processes = multiprocessing.cpu_count()

# Create a pool of processes to download the images
pool = multiprocessing.Pool(num_processes)
results = []
with tqdm(total=len(url_list)) as pbar:
    for file_name in pool.imap_unordered(download_image, url_list):
        if file_name is not None:
            results.append(file_name)
        pbar.update()

pool.close()
pool.join()

print(f"Downloaded {len(results)} files")
