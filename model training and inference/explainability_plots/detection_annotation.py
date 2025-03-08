import cv2
import torch
import argparse
import numpy as np
import os

def annotate_heatmap(heatmap_path, bbox_coords, conf, output_path="annotated_eigencam.png"):
    """
    Draws a bounding box with confidence score on the heatmap.

    Args:
        heatmap_path (str): Path to the EigenCAM heatmap image.
        bbox_coords (list): Bounding box coordinates in [x_min, y_min, x_max, y_max] format.
        conf (float): Confidence score of the bounding box.
        output_path (str): Path to save the annotated heatmap.
    """
    heatmap = cv2.imread(heatmap_path)
    if heatmap is None:
        raise ValueError(f"Error: Could not load heatmap from {heatmap_path}")

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    bbox = torch.tensor(bbox_coords).int().tolist() 
    cv2.rectangle(heatmap_resized, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    text = f"conf:{conf:.2f}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    
    text_x = bbox[0]
    text_y = bbox[1] + text_size[1] + 5  

    cv2.putText(heatmap_resized, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    final_output = cv2.resize(heatmap_resized, (640, 640))

    output_file = os.path.join(output_path, "annotated_eigencam.png")
    cv2.imwrite(output_file, final_output)
    print(f"Annotated image saved as {output_file}")

    cv2.imshow("Annotated Heatmap", final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate EigenCAM heatmap with bounding box and confidence score.")
    
    parser.add_argument("--heatmap_path", type=str, required=True, help="Path to the EigenCAM heatmap image.")
    parser.add_argument("--bbox", type=float, nargs=4, required=True, help="Bounding box coordinates [x_min, y_min, x_max, y_max].")
    parser.add_argument("--conf", type=float, required=True, help="Confidence score of the bounding box.")
    parser.add_argument("--output_path", type=str, default="annotated_eigencam.png", help="Path to save the annotated heatmap.")

    args = parser.parse_args()

    annotate_heatmap(args.heatmap_path, args.bbox, args.conf, args.output_path)
