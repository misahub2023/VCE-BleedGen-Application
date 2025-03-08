# Training and Inference of Segmentation and Detection Models

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/misahub2023/VCE-BleedGen-Application.git  
   cd VCE-BleedGen-Application/model\ training\ and\ inference
   ```
2. Install requirements with Python 3.11.6:
   ```bash
   pip install -r requirements.txt
   ```
   
## Dataset Structure
+ The dataset structure were as follows:
+ datasets/
    + WCEBleedGen/
        + bleeding/
            + images/  
            + bounding_boxes/
                + YOLO-TXT/
                + TXT/
            + annotations/
        + non-bleeding/
            + images/
            + annotations/

## Training and Inference Scripts
### Segmentation

- **Models**: The following segmentation models were trained:
  - **UNet**
  - **SegNet**
  - **LinkNet**
  
- **Hyperparameters**:
  - **Learning Rate**: `1e-4`
  - **Epochs**: `250`
  - **Batch Size**: `32`
  - **Optimizer**: Adam (`tf.keras.optimizers.Adam`)
  - **Loss Function**: Binary Crossentropy
  - **Metrics**:
    - Accuracy (`acc`)
    - Recall (`tf.keras.metrics.Recall()`)
    - Precision (`tf.keras.metrics.Precision()`)
    - Intersection-over-Union (IoU)
    - Dice Coefficient

- **Training Process**:
  - Training runs for 250 epochs.
  - The model is compiled with the Adam optimizer and evaluated using accuracy, recall, precision, IoU, and Dice coefficient.
  - A CSV logger (`CSVLogger`) is used to store training history.
  - Model weights are saved as `"model_name.h5"`.
#### Scripts Description and Usage:
1. data_loader.py: 
   This script includes functions for loading and preprocessing the dataset for segmentation training. It organizes images and annotations, splits them into training and validation sets, and creates TensorFlow datasets with image resizing, normalization, and batching for efficient model training.
2. linknet.py, segnet.py and unet.py defines the architectures for the respective models.
3. utils.py: This script defines evaluation metrics and loss functions for segmentation models. It includes the Dice Coefficient, which measures overlap between predicted and ground truth masks, Dice Loss for optimization, and Intersection over Union (IoU) to assess model performance.
4. training.py:This script trains segmentation models (**UNet, SegNet, or LinkNet**) using **binary cross-entropy loss** and **Adam optimizer**, logging training progress. It supports custom learning rates, epochs, and batch sizes.

    ##### Usage:  
    ```bash
    python train.py --model unet --lr 1e-4 --epochs 250 --batch 32
    ```
5. visualization.py: This script loads a trained segmentation model, applies it to an input image, and visualizes the predicted segmentation mask alongside the original image and ground truth mask. The output includes a **segmentation overlay** where detected regions are highlighted in green.

    ##### Usage:  
    ```bash
    python infer.py --model_weights unet_test.h5 --image_path sample.jpg --mask_path sample_mask.jpg
    ```  
    This will display the **original image, true mask, predicted mask, and overlay visualization**.
   ##### Example:
   with segnet model
   ![image](https://github.com/user-attachments/assets/b3b7b48b-a8d7-4435-af92-72a940721e92)
### Detection
- **Models**: The following detection models were trained:
  - **Yolov5nu**
  - **YoloV12n**
  - **FasterRCNN**

#### Hyperparameters
##### **YOLOv5nu and YOLOv12**  
  The following hyperparameters were used for both **YOLOv5nu** and **YOLOv12** object detection models:
  
  - **Model:** `yolov5nu.pt` / `yolo12n.pt`
  - **Dataset:** `custom_data1.yaml`
  - **Epochs:** `100`
  - **Batch Size:** `16`
  - **Image Size:** `640`
  - **Optimizer:** `auto`
  - **Learning Rate (lr0):** `0.01`
  - **Final Learning Rate Factor (lrf):** `0.01`
  - **Momentum:** `0.937`
  - **Weight Decay:** `0.0005`
  - **Warmup Epochs:** `3.0`
  - **Warmup Momentum:** `0.8`
  - **Warmup Bias Learning Rate:** `0.1`
  - **IOU Threshold:** `0.7`
  - **Max Detections:** `300`
  - **Augmentations:**
    - **Flip Left-Right:** `0.5`
    - **Mosaic Augmentation:** `1.0`
    - **Auto Augment:** `randaugment`
    - **HSV Hue:** `0.015`
    - **HSV Saturation:** `0.7`
    - **HSV Value:** `0.4`
    - **Scale:** `0.5`
    - **Translate:** `0.1`
    - **Shear:** `0.0`
    - **Perspective:** `0.0`
    - **Mixup:** `0.0`
    - **Copy-Paste Augmentation:** `0.0`
  - **Training Settings:**
    - **Patience:** `100`
    - **Workers:** `8`
    - **Pretrained:** `True`
    - **Overlap Mask:** `True`
    - **Mask Ratio:** `4`
    - **Dropout:** `0.0`
    - **Validation Split:** `val`
    - **NMS:** `False`

##### **Faster R-CNN ResNet50 FPN v2**
The **Faster R-CNN** model was trained with the following hyperparameters:

  - **Model:** `FasterRCNN_ResNet50_FPN_V2`
  - **Pretrained:** `True`
  - **Number of Classes:** `2` (Bleeding and Background)
  - **Optimizer:** `Adam`
    - **Learning Rate:** `1e-4`
  - **Learning Rate Scheduler:** `StepLR`
    - **Step Size:** `5`
    - **Gamma:** `0.1`
  - **Backbone Weights:** `ResNet50_Weights`
  - **Trainable Backbone Layers:** `3` (default)
  - **Progress Bar Enabled:** `True`
#### Scripts Description and Usage:
1. dataprep.py: This script splits a dataset of **bleeding** and **non-bleeding** images into **train** and **validation** sets, while preserving corresponding **YOLO** and **VOC** label formats. The dataset is shuffled and split **80-20** before copying files into structured output directories.  

    ##### **Usage:**  
    ```bash
    python split_dataset.py --base_path /path/to/dataset --output_path /path/to/output
    ```
    Replace `/path/to/dataset` with the source dataset directory and `/path/to/output` with the target directory for the split dataset.

2. yaml_prep.py: 
This script generates a **YOLO dataset configuration YAML file**, specifying the dataset path and train/validation image directories. The generated file is used for training YOLO models.  

   ##### **Usage:**  
   ```bash
   python yaml_prep.py --output_path /path/to/save --dataset_path /path/to/dataset
   ```
   Replace `/path/to/save` with the directory where the YAML file should be saved and `/path/to/dataset` with the base dataset directory.
3. yolo_training.py:
This script trains a **YOLOv5nu or YOLOv12n model** on a custom dataset using the Ultralytics YOLO framework. It loads the specified model, trains it for a given number of epochs, and saves the results.  

   ##### **Usage:**  
   ```bash
   python yolo_training.py --model_name yolov5nu --data_path /path/to/custom_data1.yaml --epochs 100
   ```
4. yolo_inference.py:
   This script performs object detection on an input image using a trained YOLO model. It loads the model, runs inference, and displays the image with detected bounding boxes.
   ##### **Usage:**
   ```bash
   python yolo_inference.py --model_path yolov5nu.pt --image_path /path/to/image.jpg
   ```
#### Fasterrcnn scripts:
1. data_loader.py:
This script loads the **Wireless Capsule Endoscopy (WCE) dataset** for object detection using PyTorch. It reads images and Pascal VOC format bounding box annotations, applies transformations, and prepares batches for model training.  
   ##### **Usage:**  
   ```bash
   python data_loader.py --dataset_root /path/to/dataset --split train
   ```  
   Replace `/path/to/dataset` with the dataset's root directory and choose `train` or `val` for the dataset split.
2. training.py: 
This script trains a **Faster R-CNN** model using the **Wireless Capsule Endoscopy (WCE) bleeding dataset**. It logs training loss, validation loss, and mAP@0.5 while saving the best model based on validation performance.  

   ##### **Usage:**  
   ```bash
   python training.py --dataset_root /path/to/dataset --batch_size 4 --epochs 100 --lr 1e-4
   ```  
   Replace `/path/to/dataset` with the dataset's root directory and adjust hyperparameters as needed.
3. inference.py:
This script loads a trained **Faster R-CNN** model and runs inference on a given image. It processes the image using **Torchvision transforms** and outputs raw model predictions.  

   ##### **Usage:**  
   ```bash
   python inference.py --model_path best.pth --image_path sample.jpg
   ```  
   Replace `best.pth` with the path to your trained model and `sample.jpg` with the image file for inference.

## Explainability Plots

For all models Eigencam plots were generated using the deepest convolutional layer of the respective architectures.
### scripts:
1. eigencam_torch.py:
   This script contains the standard torch implementation of the eigencam plot
2. eigencam_tensorflow.py:
   This script contains the standard tensorflow implementation of the eigencam plot.
3. yolo_modified_eigencam_torch.py:
   This script contains the modified implementation of eigencam in pytorch to make it compatible with the yolo models.
4. gen_heatmap_segmentation.py:
This script loads a **segmentation model**, applies **EigenCAM** to generate a heatmap for a given layer, and overlays the segmentation mask on the input image. The final overlay is saved as `overlay.png`.  

   ##### **Usage:**  
   ```bash
   python gen_heatmap_segmentation.py --model_name unet --weights_path model_weights.h5 \
   --output_path ./results --layer_name conv5_block3_out --image_path sample.jpg
   ```  
   Replace `unet` with your model name, `model_weights.h5` with the trained weights, and `sample.jpg` with the image file for visualization.
5. gen_heatmap_yolo.py:
This script applies **EigenCAM** to a YOLO model, generating a **heatmap overlay** on an input image for a specific model layer. The result is displayed and saved as `overlay.png`.  

   ##### **Usage:**  
   ```bash
   python gen_heatmap_yolo.py --model_path yolov5nu --image_path sample.jpg --layer_name model.23
   ```  
   Replace `yolov5nu` with the chosen model, `sample.jpg` with the input image, and `model.23` with the target layer for visualization.

6. gen_heatmap_fasterrcnn.py:
This script applies **EigenCAM** to a Fastercnn model, generating a **heatmap overlay** for a specific model layer. The heatmap is displayed and saved to a specified output path.  

   ##### **Usage:**  
   ```bash
   python gen_heatmap_fasterrcnn.py --model_path yolov5nu --image_path sample.jpg --layer_name model.23 --output_path results/
   ```  
   - `--model_path` → Choose between `yolov5nu` or `yolo12n`.  
   - `--image_path` → Path to the image for visualization.  
   - `--layer_name` → Target layer name for EigenCAM visualization.  
   - `--output_path` → Folder where `overlay.png` will be saved.

7. detection_annotation.py:
This script overlays **bounding boxes and confidence scores** on EigenCAM heatmaps to visualize object detection results.

   ##### **Usage:**  
   ```bash
   python annotate_eigencam.py --heatmap_path heatmap.png --bbox 50 60 150 180 --conf 0.92 --output_path results/
   ```  
   - `--heatmap_path` → Path to the **EigenCAM heatmap** image.  
   - `--bbox` → Bounding box coordinates `[x_min, y_min, x_max, y_max]`.  
   - `--conf` → Confidence score of the detected object.  
   - `--output_path` → Directory to save the **annotated heatmap** as `annotated_eigencam.png`.
8. segmentation_annotation.py:
The function in the script draws segmentation mask boundaries on an overlay image, highlighting detected regions.






