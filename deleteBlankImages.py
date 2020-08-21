from PIL import Image
import os
train_images_path = '../Data/MassachusettsRoads/Train/Images/samples'
masks_path = '../Data/MassachusettsRoads/Train/Masks/samples'

for file in os.listdir(train_images_path):
    current_file = os.path.join(train_images_path, file)
    img = Image.open(current_file)
    clrs = img.getcolors()
    if(len(clrs) <= 2):
        print(file)

