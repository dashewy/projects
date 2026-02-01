import tensorflow as tf
import numpy as np
from rembg import remove
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image

def image_predictor(image_path):
    
    rotten = load_model('rotten_model.keras')
    
    labels = ['Fresh', 'Rotten']
    
    image_height = 256
    image_width = 256
    
    img = image.load_img(image_path, target_size=(256, 256))
    # removes background
    img = remove(img)
    # white background fix
    white_bg = Image.new("RGB", img.size, (256, 256, 256))
    
    white_bg.paste(img, mask=img.split()[3])

    img = white_bg.resize((256, 256))
        
    plt.imshow(img)
    plt.axis('off')
    
    img_array = image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.
    
    input_arr = np.expand_dims(img_array, axis=0)
    
    predictions = rotten.predict(input_arr)
    print(predictions)
    
    pred_prob = predictions[0][0]
    pred_label = labels[int(pred_prob > 0.5)]
    
    plt.title(f'Your fruit is {pred_label}')
    plt.show()
    
    return pred_label

# test
print(image_predictor('/Users/alex/data_prac/apple_2.jpeg'))