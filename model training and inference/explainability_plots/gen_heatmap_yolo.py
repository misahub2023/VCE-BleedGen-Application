from ultralytics import YOLO
import argparse
from yolo_modified_eigencam_torch import EigenCAM
import cv2

def main(model_path, image_path, layer_name):
    model = YOLO(model_path)
    map=EigenCAM(model, device='cuda', layer_name=layer_name)
    out,overlay=map.get_heatmap(image_path)
    cv2.imshow(overlay)
    cv2.imwrite("overlay.png", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(out[0].boxes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, choices=["yolov5nu", "yolo12n"], help="YOLO model to use")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file") 
    parser.add_argument("--layer_name", type=str, required=True, help="Name of the layer to visualize")
    
    args = parser.parse_args()
    main(args.model_path, args.image_path, args.layer_name)

