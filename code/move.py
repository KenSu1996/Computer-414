import os
import shutil
import random

# Set source directory path
source_dir = './res_aligned'

# Set target directory path
train_dir = './res_aligned/train'
val_dir = './res_aligned/val'

# make sure target director exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# get all files
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Randomly shuffle the file list
random.shuffle(files)

# Split training and validation sets (e.g., 80% training, 20% validation)
split_index = int(0.8 * len(files))
train_files = files[:split_index]
val_files = files[split_index:]

#move file to folder
for f in train_files:
    shutil.move(os.path.join(source_dir, f), os.path.join(train_dir, f))

for f in val_files:
    shutil.move(os.path.join(source_dir, f), os.path.join(val_dir, f))
