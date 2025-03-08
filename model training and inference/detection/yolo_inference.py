import argparse
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def visualize_inference(model_path, image_path):
    model = YOLO(model_path)
    results = model(image_path)
    
    for result in results:
        img = result.plot()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained YOLO model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    
    args = parser.parse_args()
    visualize_inference(args.model_path, args.image_path)
