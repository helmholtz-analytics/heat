# Q1: What is the purpose of the MemmapReader?


# Q2: Why do we apply transforms in the Reader?


# Q3: How does the DataLoader interact with the MemmapReader?


# Q4: Can MemmapReader work with num_workers in DataLoader?


# Q5: How do we know the memmap dataset is working correctly?


import sys

sys.path.append("./hpc_pytorch_loader")
from datasets.memmap.memmap_reader import MemmapReader
from torch.utils.data import DataLoader
from torchvision import transforms

# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Initialize the reader
reader = MemmapReader(dataset_path="./memmap_dataset", transform=transform)

# Create a DataLoader
dataloader = DataLoader(reader, batch_size=32, num_workers=4)

# Iterate through the dataset
for i, (images, labels) in enumerate(dataloader):
    # Your processing code here
    print("i: ", i)
    if i > 100:
        break
