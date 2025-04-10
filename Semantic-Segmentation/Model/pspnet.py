# -*- coding: utf-8 -*-
"""m23csa526 (Apr 8, 2025, 10:36:42 PM)

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/embedded/projects/cv-project-456217/locations/us-central1/repositories/e8b840b2-98a5-480b-a294-0ff7ce6d938a
"""

import kagglehub
import shutil
import os
dataset_path = kagglehub.dataset_download('yaroslavchyrko/rescuenet')

print('Data source import complete.')

dataset_path

import shutil
import os
# Define destination
destination_path = "/content/rescuenet"

# Move dataset to /content
shutil.move(dataset_path, destination_path)

!pip install segmentation-models

# Remove standalone Keras if it’s installed (which breaks compatibility)
!pip uninstall -y keras

# Reinstall TensorFlow which includes tf.keras
!pip install -U tensorflow

!pip install -U git+https://github.com/qubvel/segmentation_models.git

import tensorflow as tf
import segmentation_models as sm

# Set framework to use TensorFlow's Keras
sm.set_framework('tf.keras')
print("Framework set to:", sm.framework())

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import segmentation_models as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sm.set_framework('tf.keras')
print(sm.framework())  # should print 'tf.keras'


# --- Dataset Generator ---
class RescuenetDataset(Sequence):
    def __init__(self, image_dir, mask_dir, image_ids, batch_size=8, img_size=(256, 256), num_classes=12):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = image_ids
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []

        for img_id in batch_ids:
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            mask_path = os.path.join(self.mask_dir, f"{img_id}_lab.png")

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            img = cv2.resize(img, self.img_size)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

            img = img.astype(np.float32) / 255.0
            mask = tf.keras.utils.to_categorical(mask, num_classes=self.num_classes)

            images.append(img)
            masks.append(mask)

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

# --- Prepare Dataset ---
image_dir = '/content/rescuenet/RescueNet/train/train-org-img'
mask_dir = '/content/rescuenet/RescueNet/train/train-label-img'

val_image_dir = '/content/rescuenet/RescueNet/val/val-org-img'
val_mask_dir = '/content/rescuenet/RescueNet/val/val-label-img'

train_ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]
val_ids = [os.path.splitext(f)[0] for f in os.listdir(val_image_dir) if f.endswith('.jpg')]
# train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

train_gen = RescuenetDataset(image_dir, mask_dir, train_ids,img_size=(384,384))
val_gen = RescuenetDataset(val_image_dir, val_mask_dir, val_ids,img_size=(384,384))

# --- Build DeepLabV3+ Model ---
sm.set_framework('tf.keras')
sm.framework()

# model = sm.DeepLabV3Plus(
#     backbone_name='resnet50',
#     encoder_weights='imagenet',
#     classes=12,
#     activation='softmax'
# )

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
# define model
model = sm.PSPNet(BACKBONE, encoder_weights='imagenet',classes=12, activation='softmax')

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=sm.losses.categorical_crossentropy,
    metrics=[sm.metrics.iou_score]
)

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_cb = ModelCheckpoint(
    "/content/pspnet_rescuenet.h5",              # save to this file
    monitor="val_iou_score",      # or "val_loss"
    mode="max",                   # maximize IoU
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
# --- Train Model ---

history = model.fit(train_gen, validation_data=val_gen, epochs=20,callbacks=[checkpoint_cb])

# --- Save Model ---
model.save('pspnet_rescuenet2.h5')

# --- Predict on Sample ---
sample_img, sample_mask = val_gen[0]
pred = model.predict(np.expand_dims(sample_img[0], axis=0))
pred_mask = np.argmax(pred[0], axis=-1)
true_mask = np.argmax(sample_mask[0], axis=-1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(sample_img[0])

plt.subplot(1, 3, 2)
plt.title("True Mask")
plt.imshow(true_mask)

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(pred_mask)
plt.tight_layout()
plt.show()
