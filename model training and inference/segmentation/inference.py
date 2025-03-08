import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segmentation.training import get_model

def preps_image(image_path, size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main(model_name, weights_path, image_path, output_path, size, threshold):
    model = get_model(model_name)
    model.load_weights(weights_path)
    img = preps_image(image_path, size)
    output_mask = model(img)
    output_mask = (output_mask.numpy() > threshold).astype(np.uint8)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.imread(image_path)[:, :, ::-1])
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(output_mask), cmap="gray")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output.png")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.model_name, args.weights_path, args.image_path, args.output_path, args.size, args.threshold)
