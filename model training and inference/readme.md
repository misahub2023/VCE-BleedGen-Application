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

## Training Scripts
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

2. 

  





