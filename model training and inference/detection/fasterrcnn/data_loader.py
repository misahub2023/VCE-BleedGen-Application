import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np
import argparse

def parse_voc_annotations(annotation_path):
    """
    Parses Pascal VOC annotations from a TXT file.
    """
    boxes = []
    labels = []
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 4: 
                x_min, y_min, x_max, y_max = map(float, parts)
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)  
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

class WCEBleedGenDataset(Dataset):
    def __init__(self, root, split="train", transform=None, mask_transform=None):
        """
        Args:
            root (str): Root directory of the dataset.
            split (str): One of ['train', 'val'].
            transform (callable, optional): Transform to apply to the images.
            mask_transform (callable, optional): Transform to apply to the segmentation masks.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = []

        image_dir = os.path.join(root, "images", split)
        mask_dir = os.path.join(root, "labels_voc", split)

        for img_name in os.listdir(image_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, img_name)
                mask_path = os.path.join(mask_dir, img_name.replace(".png", ".txt"))
                self.samples.append({"image": image_path, "mask": mask_path})

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        annotation_path = sample["mask"]

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        boxes, labels = parse_voc_annotations(annotation_path)
        target = {"boxes": boxes, "labels": labels}

        return image, target

def collate_fn(batch):
    """
    Custom collate function for batching images and targets.
    """
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train", help="Dataset split")
    args = parser.parse_args()

    transform = T.Compose([T.ToTensor()])
    dataset = WCEBleedGenDataset(root=args.dataset_root, split=args.split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    for images, targets in dataloader:
        print("Images batch shape:", images.shape)
        print("Detection targets:", targets)
        break
