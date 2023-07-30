from PIL import Image
import os
from libtiff import TIFF
import staintools
import tqdm
from concurrent.futures import ThreadPoolExecutor

target = staintools.read_image("/Data/test/72e40acccadf.tif")
target = staintools.LuminosityStandardizer.standardize(target)

# Stain normalize
normalizer = staintools.StainNormalizer(method="vahadane")
normalizer.fit(target)


def apply_transformation(image):
    # Implement your own image transformation logic here
    # Example: Apply stain normalization
    transformed_image = normalizer.transform(image)
    return transformed_image


def transform_image(image_path, output_folder):
    filename = os.path.basename(image_path)
    image = TIFF.open(image_path).read_image()
    # Apply the transformation
    transformed_image = apply_transformation(image)

    # Save the transformed image in the output folder
    output_path = os.path.join(output_folder, filename)
    Image.fromarray(transformed_image).save(output_path)

    # print(f"Transformed image saved: {output_path}")


def transform_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of image paths in the input folder
    image_paths = [
        os.path.join(input_folder, filename)
        for filename in os.listdir(input_folder)
        if filename.endswith((".tif", ".tiff"))
    ]

    with ThreadPoolExecutor() as executor:
        # Iterate through all image paths and submit them to the executor
        futures = [
            executor.submit(transform_image, image_path, output_folder)
            for image_path in image_paths
        ]

        # Wait for all tasks to complete
        for future in tqdm.tqdm(
            futures, total=len(futures), desc="Transforming Images"
        ):
            future.result()


# Usage example
input_folder = "/Data/train"
output_folder = "/Data/train_stained"

transform_images(input_folder, output_folder)
