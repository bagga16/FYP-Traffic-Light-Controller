import os
import random
import shutil

# Paths
dataset_path = "dataset"
train_images_path = os.path.join(dataset_path, "train/images")
train_labels_path = os.path.join(dataset_path, "train/labels")
val_images_path = os.path.join(dataset_path, "val/images")
val_labels_path = os.path.join(dataset_path, "val/labels")

# Create val directories if they don't exist
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

# Get all training images
images = os.listdir(train_images_path)
random.shuffle(images)

# Split: 80% train, 20% val
val_split = int(len(images) * 0.2)
val_images = images[:val_split]

# Move validation images and labels
for img in val_images:
    # Move image
    shutil.move(os.path.join(train_images_path, img), os.path.join(val_images_path, img))

    # Move corresponding label
    label_file = img.replace('.jpg', '.txt').replace('.png', '.txt')
    shutil.move(os.path.join(train_labels_path, label_file), os.path.join(val_labels_path, label_file))

print(f"Validation set created with {len(val_images)} images.")
