# Q1: What is the purpose of the MemmapConverter?

# Q2: Why apply ResizeAndConvert before converting?

# Q3: What does images_per_file control?

# Q4: Why specify batch_size and num_workers in the converter?

# Q5: What does shape=(28, 28) and im_mode="RGB" do?

# Q6: What happens when converter.convert() is called?


import sys
sys.path.append("./hpc_pytorch_loader")
from datasets.memmap.memmap_converter import MemmapConverter
from utils.utils import ResizeAndConvert
from torchvision.datasets import CIFAR10


# Prepare the dataset
dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=ResizeAndConvert((32, 32), "RGB")
)

# Initialize the converter
converter = MemmapConverter(
    input_data=dataset,
    output_path="./memmap_dataset",
    images_per_file=10000,
    batch_size=1000,
    num_workers=4,
    shape=(32, 32),
    im_mode="RGB"
)

# Convert the dataset
converter.convert()
