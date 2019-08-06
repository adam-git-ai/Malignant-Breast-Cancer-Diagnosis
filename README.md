# Malignant-Breast-Cancer-Diagnosis
A Keras-based Neural Network which predicts whether a breast cancer tumour is malignant or benign.

# Summary
A neural network built with Keras consisting of two hidden layers. ReLU and Sigmoid activation are used.
The model is trained and validated using a data set (n = 569) from Wisconsin Breast Cancer Database; data collected by by Dr. William H. Goldberg, W. Nick Street and Olvi L. Mangasarian at the University of Wisconsin. The data consists of 30 features calculated from digitized images of Fine Needle Aspirate (FNA) of breast mass. The data is thoroughly described in the links below.

https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/version/2
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

# Use
Clone the data to the same directory as the scripts from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/downloads/breast-cancer-wisconsin-data.zip/2.

The data is preprocessed by ´´´dataprocessing.py´´´. As default, 80 % of the instances are used for training of the algorithm, the remaining 20 % is used to evaluate the model's accuracy. This ratio can be altered through changing the parameter ´´´numerator´´´.

The script ´´´neural.py´´´ is responsible for training and evaluating the neural network.

# Prerequisites
NumPy
Tensorflow
Keras, installation: https://keras.io/#installation
