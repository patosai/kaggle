#!/usr/bin/env python3

import csv
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random
from sklearn.ensemble import RandomForestClassifier


def read_file(filename):
    print(f'Reading file {filename}')
    data = []
    with open(filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            data.append(row)
    return data


def split_train_validation(data, split_percentage=0.8):
    print(f'Splitting train/validation set')
    data_length = len(data)
    index_to_split = int(data_length * split_percentage)
    random.seed()
    random.shuffle(data)
    return data[:index_to_split], data[index_to_split:]


def extract_features(data, extract_labels=True):
    print(f'Extracting features')
    features = []
    labels = []
    name_regex = re.compile(r' ([A-Za-z]+)\.')
    for row in data:
        sex_classification = {"male": 0, "female": 1}[row["Sex"]]
        age_classification = -1
        if row["Age"]:
            # classification buckets created using pandas.qcut(data["Age"], 4)
            # split training data equally into 4 parts
            age = float(row["Age"])
            if age < 20.125:
                age_classification = 0
            elif age < 28:
                age_classification = 1
            elif age < 38:
                age_classification = 2
            elif age < 80:
                age_classification = 3

        fare_classification = -1
        if row["Fare"]:
            # classification buckets created using a similar fashion to age
            fare = float(row["Fare"])
            if fare < 7.91:
                fare_classification = 0
            elif fare < 14.454:
                fare_classification = 1
            elif fare < 31:
                fare_classification = 2
            else:
                fare_classification = 3

        embark_classification = 0
        if row["Embarked"]:
            embark_classification = {"S": 1, "C": 2, "Q": 3}.get(row["Embarked"])

        family_size = 1
        if row["SibSp"]:
            family_size += int(row["SibSp"])
        if row["Parch"]:
            family_size += int(row["Parch"])

        title_classification = -1
        if row["Name"]:
            regex_match = name_regex.search(row["Name"])
            if regex_match:
                title = regex_match.group(1)
                new_title_classification = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3}.get(title)
                if new_title_classification is not None:
                    title_classification = new_title_classification

        feature_row = [
            row["Pclass"],
            sex_classification,
            age_classification,
            fare_classification,
            embark_classification,
            family_size,
            title_classification
        ]
        features.append(feature_row)
        if extract_labels:
            labels.append(int(row["Survived"]))
    return features, labels


def train_classifier(features, labels):
    print(f'Training')
    labels_one_hot = np.identity(2)[labels]
    rf = RandomForestClassifier(n_estimators=256, random_state=42)
    rf.fit(features, labels_one_hot)
    return rf


def predict_classification(classifier, features):
    print(f'Predicting')
    prediction = np.asarray(classifier.predict(features), dtype=int)
    # return classification, not one-hot
    return prediction[:, 1]


def run_train_with_validation():
    all_train_data = read_file("data/train.csv")
    train_data, validation_data = split_train_validation(all_train_data, split_percentage=0.8)

    train_features, train_labels = extract_features(train_data)

    classifier = train_classifier(train_features, train_labels)

    validation_features, validation_labels = extract_features(validation_data)

    validation_results = predict_classification(classifier, validation_features)
    validation_accuracy = (validation_labels == validation_results).mean()
    print(f'Validation accuracy: {validation_accuracy:.5f}')
    return classifier


def run_train():
    print("Running train on full dataset")
    train_data = read_file("data/train.csv")
    train_features, train_labels = extract_features(train_data)
    return train_classifier(train_features, train_labels)


def create_test_predictions():
    print("Creating test predictions")
    classifier = run_train()
    test_data = read_file("data/test.csv")
    test_features, _ = extract_features(test_data, extract_labels=False)
    predictions = predict_classification(classifier, test_features)

    out_filename = "predictions.csv"
    print(f"Writing to {out_filename}")
    with open(out_filename, 'w') as csv_file:
        fieldnames = ['PassengerId', 'Survived']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row, prediction in zip(test_data, predictions):
            writer.writerow({"PassengerId": row["PassengerId"], "Survived": prediction})


#classifier = run_train_with_validation()
create_test_predictions()