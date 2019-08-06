import dataprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy

def neuralnet(features_for_training, true_class_for_training, features_for_validation, true_class_for_validation):

    # Defining layers of the neural network
    model = Sequential()
    model.add(Dense(24, input_dim = len(features_for_training[0]), activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid')) # Using sigmoid to get a value in the continuous interval, 0 to 1

    # Compiling the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Training the model
    model.fit(features_for_training, true_class_for_training, epochs=1000, batch_size=10)
    accuracy = model.evaluate(features_for_validation, true_class_for_validation)
    print(model.metrics_names[1], accuracy[1] * 100)


# Generating training and validation data
features_for_training, true_class_for_training, features_for_validation, true_class_for_validation = dataprocessing.gen_data()
neuralnet(features_for_training, true_class_for_training, features_for_validation, true_class_for_validation)