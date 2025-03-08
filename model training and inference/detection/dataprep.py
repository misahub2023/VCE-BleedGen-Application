import os
import shutil
import random
import argparse

def split_dataset(base_path, output_path):
    bleeding_image_path = os.path.join(base_path, 'bleeding/Images')
    bleeding_yolo_label_path = os.path.join(base_path, 'bleeding/Bounding boxes/YOLO_TXT')
    bleeding_voc_label_path = os.path.join(base_path, 'bleeding/Bounding boxes/TXT')
    non_bleeding_image_path = os.path.join(base_path, 'non-bleeding/Images')

    output_image_path = os.path.join(output_path, 'images')
    output_yolo_label_path = os.path.join(output_path, 'labels_yolo')
    output_voc_label_path = os.path.join(output_path, 'labels_voc')

    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_image_path, split), exist_ok=True)
        os.makedirs(os.path.join(output_yolo_label_path, split), exist_ok=True)
        os.makedirs(os.path.join(output_voc_label_path, split), exist_ok=True)

    bleeding_images = [f for f in os.listdir(bleeding_image_path) if f.endswith('.png')]
    non_bleeding_images = [f for f in os.listdir(non_bleeding_image_path) if f.endswith('.png')]

    all_images = [(file, "bleeding") for file in bleeding_images] + [(file, "non-bleeding") for file in non_bleeding_images]
    random.seed(42)
    random.shuffle(all_images)

    total_images = len(all_images)
    train_split = int(0.8 * total_images)
    val_split = total_images - train_split

    train_files = all_images[:train_split]
    val_files = all_images[train_split:]

    for split, files in zip(['train', 'val'], [train_files, val_files]):
        for file, category in files:
            if category == "bleeding":
                source_image = os.path.join(bleeding_image_path, file)
                source_yolo_label = os.path.join(bleeding_yolo_label_path, file.replace('.png', '.txt'))
                source_voc_label = os.path.join(bleeding_voc_label_path, file.replace('.png', '.txt'))
            else:
                source_image = os.path.join(non_bleeding_image_path, file)
                source_yolo_label = None
                source_voc_label = None

            destination_image = os.path.join(output_image_path, split, file)
            shutil.copyfile(source_image, destination_image)

            destination_yolo_label = os.path.join(output_yolo_label_path, split, file.replace('.png', '.txt'))
            destination_voc_label = os.path.join(output_voc_label_path, split, file.replace('.png', '.txt'))

            if source_yolo_label and os.path.exists(source_yolo_label):
                shutil.copyfile(source_yolo_label, destination_yolo_label)
            else:
                open(destination_yolo_label, 'w').close()

            if source_voc_label and os.path.exists(source_voc_label):
                shutil.copyfile(source_voc_label, destination_voc_label)
            else:
                open(destination_voc_label, 'w').close()

    print("Data split and copied successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="Base dataset path")
    parser.add_argument("--output_path", type=str, required=True, help="Output dataset path")
    args = parser.parse_args()

    split_dataset(args.base_path, args.output_path)
