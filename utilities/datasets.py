from torch.utils.data import DataLoader
from modules.configs import Configs
from torchvision import transforms
from glob import glob
import torch
import PIL

configs = Configs()

# Defining dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir:str, transform) -> None:
        # Global values
        self.files_addr = list()
        self.transform = transform

        # Reading files
        for file_address in glob(f"{root_dir}/*"):
            self.files_addr.append(file_address)

    def __len__(self):
        return len(self.files_addr)
    
    def __getitem__(self, idx):
        path = self.files_addr[idx]
        image = PIL.Image.open(path)
        image = image.convert("RGB")
        image = self.transform(image)
        return image

# Creatings transforms
content_image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
style_image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Creating datasets
content_dataset = Dataset(root_dir=configs.content_images_directory, transform=content_image_transform)
style_dataset = Dataset(root_dir=configs.style_images_directory, transform=style_image_transform)

# Creating data loaders
content_dataloader = DataLoader(
    dataset=content_dataset,
    batch_size=configs.content_batch_size,
    shuffle=True,
    num_workers=configs.content_numworkers,
    drop_last=True,
    pin_memory=True
)

def infinite_style_loader(dataset, batch_size, num_workers):
    while True:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        for batch in dataloader:
            yield batch