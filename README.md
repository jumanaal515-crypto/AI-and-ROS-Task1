AI and ROS Path - Task 1: Image Recognition

This repository contains the first task for the Artificial Intelligence and Robot Operating System (ROS) track, showcasing how to train and deploy an image recognition model.

Project Steps Taken

-Model Training (Teachable Machine)
-We utilized Google Teachable Machine to train a standard image classification model.
-The model was trained using two specific classes with custom datasets: Elon Musk and Donald Trump.
-Multiple image samples were uploaded for each class to train the model effectively and evaluate its performance.
-Exporting the Model
-After successfully evaluating the model's accuracy, it was exported in TensorFlow -> Keras format.
-The downloaded files included the weights file (keras_model.h5) and the text file containing class names (labels.txt).
-Deployment and Troubleshooting in Google Colab
-The model was loaded into Google Colab using a Python script.
-Compatibility Fix: Since Google Colab updates default to Keras 3, the script was optimized using a tf_keras legacy environment setup to overcome the DepthwiseConv2D layer deserialization error (ValueError).
-The model successfully processed the input test image and accurately predicted the correct class with a high confidence score of 80%.

