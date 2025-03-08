import argparse
from eigencam_torch import EigenCAM
import cv2
import os
import torch
from PIL import Image
import torchvision.transforms as T

def main(model_path, image_path, layer_name, output_path):
    model = torch.load(model_path, map_location=torch.device('cuda'))
    model.eval()
    transform = T.Compose([
    T.ToTensor(), 
    ])
    map=EigenCAM(model, device='cuda',transform=transform, layer_name=layer_name)
    img=Image.open(image_path)
    out,overlay=map.get_heatmap(img)
    overlay = cv2.resize(overlay, (640, 640))
    output_file = os.path.join(output_path, "overlay.png")
    cv2.imwrite(output_file, overlay)
    cv2.imshow(overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(out.boxes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, choices=["yolov5nu", "yolo12n"], help="YOLO model to use")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file") 
    parser.add_argument("--layer_name", type=str, required=True, help="Name of the layer to visualize")
    parser.add_argument("--output_path", type=str, default="overlay.jpg", help="Path to save output heatmap")
    
    args = parser.parse_args()
    main(args.model_path, args.image_path, args.layer_name, args.output_path)

