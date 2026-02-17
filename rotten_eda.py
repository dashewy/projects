# import data from kaggle (rotten food)
# import kagglehub
# path = kagglehub.dataset_download("abdulrafeyyashir/fresh-vs-rotten-fruit-images")
# print("Path to dataset files:", path)

import os
import matplotlib.pyplot as plt
import random
from PIL import Image

data_path = '/Users/alex/.cache/kagglehub/datasets/abdulrafeyyashir/fresh-vs-rotten-fruit-images/versions/4/Fruit Freshness Dataset/Fruit Freshness Dataset'

classes = []
labels = []
# gets all folders in path
for fruit in os.listdir(data_path):
    # joins each fruit in dataset with path
    fruit_path = os.path.join(data_path, fruit)
    # checks if directory exists
    if os.path.isdir(fruit_path):
        # gets status for each fruit
        for status in os.listdir(fruit_path):
            # joins status and path
            status_path = os.path.join(fruit_path, status)
            # checks if directory exists
            if os.path.isdir(status_path):
                # appends each class and label to lists
                classes.append(status_path)
                labels.append(f'{fruit}_{status}')
                
# look at random sample of images
# test
# print(classes)

plt.figure(figsize=(12,8))

for index, (path, label) in enumerate(zip(classes, labels)):
    img_name = random.choice(os.listdir(path))
    img = Image.open(os.path.join(path, img_name))
    # index =! 0
    plt.subplot(2, 3, index + 1)
    plt.imshow(img)
    plt.title(label)
    plt.axis("off")
    
plt.tight_layout()
plt.show()

