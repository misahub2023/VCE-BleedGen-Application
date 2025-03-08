import cv2
import numpy as np

def draw_segmentation_boundaries(overlay, output_mask, color=(0, 255, 0), thickness=0):
    """
    Draw segmentation mask boundaries on the overlay image.

    Args:
        overlay (numpy array): The RGB overlay image (224, 224, 3).
        output_mask (numpy array): The predicted mask (1, 224, 224, 1).
        color (tuple): Color for the boundary (default: Red).
        thickness (int): Thickness of the contour line.

    Returns:
        numpy array: Image with segmentation boundaries drawn.
    """
    overlay = overlay.astype(np.uint8)
    mask = output_mask.squeeze()
    mask = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay