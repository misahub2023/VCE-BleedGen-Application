import os
import argparse
from segmentation.training import get_model
from eigencam_tensorflow import EigenCAM
import cv2
import numpy as np
from segmentation_annotation import draw_segmentation_boundaries

def preps_image(image_path, size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main(model_name, weights_path, output_path, layer_name, image_path):
    img = preps_image(image_path)
    model = get_model(model_name)
    model.load_weights(weights_path)
    eg = EigenCAM(model, device='cuda', layer_name=layer_name)
    _, overlay = eg.get_heatmap(img)
    output_mask = model(img)
    output_mask = (output_mask.numpy() > 0.5).astype(np.uint8)
    overlay = draw_segmentation_boundaries(overlay, output_mask)
    overlay=cv2.resize(overlay, (640, 640))
    cv2.imwrite(os.path.join(output_path, "overlay.png"), overlay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=".", help="Folder path to save the output overlay")
    parser.add_argument("--layer_name", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    main(args.model_name, args.weights_path, args.output_path, args.layer_name, args.image_path)



