from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
from pathlib import Path



class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def load_data(data_dir,):
    import numpy as np
    def image_to_numpy(image):
        return np.array(image).astype(np.uint8)
    # more robust loading to avoid loaing non-image files
    images = [] 
    for i in list(Path(data_dir).iterdir()):
        if not i.suffix in [".jpg", ".png", ".jpeg"]:
            continue
        else:
            images.append(image_to_numpy(Image.open(i).convert("RGB")))
    # resize the images to 512 x 512
    images = [Image.fromarray(i).resize((512, 512)) for i in images]
    images = np.stack(images)
    # from B x H x W x C to B x C x H x W
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    # images = np.array(images).transpose(0, 3, 1, 2)
    assert images.shape[-1] == images.shape[-2]
    return images

    
    
from PIL import Image
from io import BytesIO

def jpeg_compress_image(image: Image.Image, quality: int = 85) -> Image.Image:
    """
    Compresses the input PIL Image object using JPEG compression and returns
    a new PIL Image object of the compressed image.
    
    :param image: PIL Image object to be compressed.
    :param quality: JPEG compression quality. Ranges from 0 to 95.
    :return: New PIL Image object of the compressed image.
    """
    compressed_image_io = BytesIO()
    image.save(compressed_image_io, 'JPEG', quality=quality)
    compressed_image_io.seek(0)  # Reset the stream position to the beginning.
    return Image.open(compressed_image_io)

