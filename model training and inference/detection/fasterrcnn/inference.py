import argparse
import cv2
import os
import torch
from PIL import Image
import torchvision.transforms as T

def main(model_path, image_path):
    model = torch.load(model_path, map_location=torch.device('cuda'))
    model.eval()
    transform = T.Compose([
    T.ToTensor(), 
    ])
    img=Image.open(image_path)
    img=transform(img)
    img=img.unsqueeze(0)    
    out = model(img)
    print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file") 
    
    args = parser.parse_args()
    main(args.model_path, args.image_path)

