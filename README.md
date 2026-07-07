# AI and ROS Path - Task 1: Image Recognition

This repository contains the first task for the Artificial Intelligence and Robot Operating System (ROS) track, showcasing how to train and deploy an image recognition model.



##  Project Steps Taken

### 1. Model Training (Teachable Machine)
* We utilized **Google Teachable Machine** to train a standard image classification model.
* The model was trained using two specific classes with custom datasets: **Elon Musk** and **Donald Trump**.
* Multiple image samples were uploaded for each class to train the model effectively and evaluate its performance.

### 2. Exporting the Model
* After successfully evaluating the model's accuracy, it was exported in **TensorFlow -> Keras** format.
* The downloaded files included the weights file (`keras_model.h5`) and the text file containing class names (`labels.txt`).

### 3. Deployment and Troubleshooting in Google Colab
* The model was loaded into **Google Colab** using a Python script.
* **Compatibility Fix:** Since Google Colab updates default to Keras 3, the script was optimized using a `tf_keras` legacy environment setup to overcome the `DepthwiseConv2D` layer deserialization error (`ValueError`).
* The model successfully processed the input test image and accurately predicted the correct class with a high confidence score of **97.49%**.


# AI and ROS Path - Task 1: Image Recognition

This repository contains the first task for the Artificial Intelligence and Robot Operating System (ROS) track, showcasing how to train and deploy an image recognition model.




##  Python Script Used

!pip install tf_keras -q
import os; os.environ["TF_USE_LEGACY_KERAS"] = "1"; import tf_keras as keras
from tf_keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("IMG_6646.JPG").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)

