import os
from glob import glob
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
import numpy as np

def load_data(path, validation_size=0.2):
    categories = ['bleeding', 'non-bleeding']
    data = []

    for category in os.listdir(path):
        image_folder = os.path.join(path, category, 'Images')
        annotation_folder = os.path.join(path, category, 'Annotations')

        images = sorted(glob(os.path.join(image_folder, "*")))
        annotations = sorted(glob(os.path.join(annotation_folder, "*")))

        category_data = list(zip(images, annotations))

        data.extend(category_data)

    total_size = len(data)
    valid_size = int(validation_size * total_size)

    train_data, valid_data = train_test_split(data, test_size=valid_size, random_state=42)

    train_x, train_y = zip(*train_data)
    valid_x, valid_y = zip(*valid_data)

    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
     path=path.decode()
     x = cv2.imread(path)
     #print(x)
     x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
     x = cv2.resize(x, (224, 224))
     x = x/255.0
     return x

def read_mask(path):
     path=path.decode()
     x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
     x = cv2.resize(x, (224, 224))
     x = x/255.0
     x = np.expand_dims(x, axis=-1)
     return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([224, 224, 3])
    y.set_shape([224, 224, 1])

    return x, y

def tf_dataset(x, y, batch=32):
    x = list(x)
    y = list(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset