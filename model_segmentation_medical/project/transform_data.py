from PIL import Image
import os
from libtiff import TIFF
import staintools
import tqdm

target = staintools.read_image("/Data/test/72e40acccadf.tif")
target = staintools.LuminosityStandardizer.standardize(target)

# Stain normalize
normalizer = staintools.StainNormalizer(method="vahadane")
normalizer.fit(target)


def apply_transformation(image):
    # Implement your own image transformation logic here
    # Example: Rotate the image by 90 degrees
    transformed_image = staintools.LuminosityStandardizer.standardize(image)
    transformed_image = normalizer.transform(transformed_image)
    return transformed_image


def transform_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in tqdm.tqdm(os.listdir(input_folder)):
        # Check if the file is an image
        if filename.endswith((".tif", ".tiff")):
            # Open the image file
            image_path = os.path.join(input_folder, filename)
            image = TIFF.open(image_path).read_image()
            # Apply the transformation
            transformed_image = apply_transformation(image)

            # Save the transformed image in the output folder
            output_path = os.path.join(output_folder, filename)
            Image.fromarray(transformed_image).save(output_path)

            # print(f"Transformed image saved: {output_path}")


# Usage example
input_folder = "/Data/train"
output_folder = "/Data/train_stained_lumin"

transform_images(input_folder, output_folder)
