import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from unet import build_model

def read_mask_(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (224, 224))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    return x

def read_image_(path):
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224, 224))
    x = x / 255.0
    return x

def display_segmentation(image_path, true_mask_path, model):
    x = read_image_(image_path)
    y_true = read_mask_(true_mask_path)
    y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
    y_pred = cv2.resize(y_pred.astype(np.uint8), (x.shape[1], x.shape[0]))
    overlay = x.copy()
    overlay[y_pred == 1] = [0, 255, 0]

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(x)
    plt.title("Original Image")

    plt.subplot(1, 4, 2)
    plt.imshow(y_true.squeeze(), cmap='gray')
    plt.title("True Mask")

    plt.subplot(1, 4, 3)
    plt.imshow(y_pred, cmap='gray')
    plt.title("Predicted Mask")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title("Segmentation Overlay")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the model weights (.h5) file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the ground truth mask")
    args = parser.parse_args()

    model = build_model()
    model.load_weights(args.model_weights)

    display_segmentation(args.image_path, args.mask_path, model)
