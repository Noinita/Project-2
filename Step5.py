# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 13:01:20 2023

@author: noini
"""
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model=load_model("Project_2.h5")

# Load an image for prediction
#image_path = 'data/Test/Medium/Crack__20180419_06_19_09,915.bmp'
image_path = 'data/Test/Large/Crack__20180419_13_29_14,846.bmp'

# Resize 
img = image.load_img(image_path, target_size=(100, 100))  
img_array = image.img_to_array(img) 
# batch dimension
img_array = np.expand_dims(img_array, axis=0)  
img_array /= 255.0  # Normalize 

# Model predictions
predictions = model.predict(img_array)

# Predicted class label
predicted_class = np.argmax(predictions)
predicted_prob = np.max(predictions)


class_names = {
    0: "Large",
    1: "Medium",
    2: "None",
    3: "Small"
}


true_label = "Large"


predicted_label = class_names[predicted_class]  
output = f"True Label: {true_label}\nPredicted Label: {predicted_label}\nProbability: {predicted_prob:.2f}"

#Image
plt.imshow(img)
plt.title(output)
plt.axis('off')
#Probabilities
for i, prob in enumerate(predictions[0]):
    label = class_names[i]
    text_color = 'pink' if label == predicted_label else 'white' 
    plt.text(5, 20 + 10 * i, f"{label}: {prob:.2f}", color=text_color, fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
plt.show()