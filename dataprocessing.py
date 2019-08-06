import csv
import os.path
import numpy as np

def make_files(numerator):

    print("Dividing data, using",(numerator*10//10)*10,"% for training...")
    trainingD = open('trainingdata.csv', 'w')
    validationD = open('validationdata.csv', 'w')
    with open('data.csv', 'r') as original:
        reader = [line.split("\n") for line in original]
        row_count = sum(1 for row in reader)
        row_limit = (numerator * row_count) // 10
        for i, row in enumerate(reader):
            if i > 0 and i <= row_limit:
                trainingD.write(row[0]+"\n")
            if i >= row_limit:
                validationD.write(row[0]+"\n")
        trainingD.close()
        validationD.close()
        original.close()
        print("Training and validation data created...")


def read_files():

    with open('trainingdata.csv', newline = '\n') as trainD:

        trainingD = csv.reader(trainD, delimiter= ',')
        features_for_training = []
        true_class_for_training = []

        for row in trainingD:
            features_for_training.append(row[2:])
            true_class_for_training.append(row[1])

    with open('validationdata.csv', newline = '\n') as validD:
        
        validationD = csv.reader(validD, delimiter= ',')
        features_for_validation = []
        true_class_for_validation = []

        for row in validationD:
            features_for_validation.append(row[2:])
            true_class_for_validation.append(row[1])
            
    print("Training and validation data loaded...")

    return features_for_training, true_class_for_training, features_for_validation, true_class_for_validation


def true_class_conversion(class_vec):   # Converting 'M' = malignant and 'B' = benign to 'M' = 1 and 'B' = 0
    returnVec = []
    for each in class_vec:
        if each == 'M':
            returnVec.append(1)
        else:
            returnVec.append(0)
    return returnVec

def type_conversion(feature_vec):

    temp = []
    for line in feature_vec:
        line = [float(x) for x in line]
        temp.append(line)

    return temp


def gen_data():

    if os.path.exists('trainingdata.csv') == False:
        numerator = 8    # Numerator, used to divide the data between training data and validation data
        make_files(numerator)
    
    features_for_training, true_class_for_training, features_for_validation, true_class_for_validation = read_files()

    true_class_for_training = true_class_conversion(true_class_for_training)
    true_class_for_validation = true_class_conversion(true_class_for_validation)

    features_for_training = type_conversion(features_for_training)
    features_for_validation = type_conversion(features_for_validation)

    features_for_training = np.array(features_for_training)
    features_for_validation = np.array(features_for_validation)

    return features_for_training, true_class_for_training, features_for_validation, true_class_for_validation