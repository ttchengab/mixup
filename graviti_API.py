from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tensorbay import GAS
from tensorbay.dataset import Segment
from tensorbay.dataset import Dataset as TensorBayDataset

class Flower17_Dataset(Dataset):
    def __init__(self, gas, segment_name, transform):
        super().__init__()
        self.dataset = TensorBayDataset("Flower17", gas)
        self.segment = self.dataset[segment_name]
        self.category_to_index = self.dataset.catalog.classification.get_category_to_index()
        self.transform = transform

    def __len__(self):
        return len(self.segment)

    def __getitem__(self, idx):
        data = self.segment[idx]
        with data.open() as fp:
            image_tensor = self.transform(Image.open(fp))

        image_tensor = transforms.Resize([224,224])(image_tensor)
        label = self.category_to_index[data.label.classification.category]
        print(image_tensor.shape)
        return image_tensor, label

def transformations():
    to_tensor = transforms.ToTensor()
    normalization = transforms.Normalize(mean = [0.485], std=[0.229])
    my_transforms = transforms.Compose([to_tensor, normalization])
    return my_transforms

ACCESS_KEY = "ACCESSKEY-2cb17e556b363c1a04a00eba1dd486d4"
gas = GAS(ACCESS_KEY)
dataset = TensorBayDataset("Flower17", gas)
# segment_names = dataset.keys()
transform = transformations()
train_dataset = Flower17_Dataset(gas, 'train', transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_dataset = Flower17_Dataset(gas, 'validation', transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

for image, label in train_dataloader:
    print(image, label)
    break
