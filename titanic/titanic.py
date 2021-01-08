#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch


def transform_row(row):
    # in train dataset, everyone has a
    # - Pclass
    # - Sex
    # - SibSp
    # - Parch
    # - Ticket
    # - Fare

    # some are missing
    # - Age: 177
    # - Cabin: 687
    # - Embarked: 2
    row["PassengerId"] = int(row["PassengerId"])
    if "Survived" in row:
        row["Survived"] = int(row["Survived"])

    # 1, 2, or 3
    row["Pclass"] = int(row["Pclass"])

    # Sex can be "male" or "female"
    row["Sex-male"] = row["Sex"] == "male"
    row["Sex-female"] = row["Sex"] == "female"

    if row["Age"]:
        row["Age"] = float(row["Age"])
    else:
        row["Age"] = 35.0

    if row["Cabin"]:
        # some in the data have multiple cabins, just take the first one
        # Titanic has cabins from A to G deck
        match = re.match(r"([A-G])(\d+)", row["Cabin"])
        if match:
            deck, number = match.groups()
            row["Cabin"] = {"deck": deck,
                            "number": int(number)}
        else:
            row.pop('Cabin')
    else:
        row.pop('Cabin')

    row["SibSp"] = int(row["SibSp"])
    row["Parch"] = int(row["Parch"])
    #row["Ticket"] = int(row["Ticket"])
    if row["Fare"]:
        row["Fare"] = float(row["Fare"])
    else:
        row["Fare"] = 0.0

    row["Embarked-Cherbourg"] = row["Embarked"] == "C"
    row["Embarked-Queenstown"] = row["Embarked"] == "Q"
    row["Embarked-Southampton"] = row["Embarked"] == "S"
    return row


def row_to_model_input(row):
    output = np.array([
        row["Pclass"] == 1,
        row["Pclass"] == 2,
        row["Pclass"] == 3,
        row["Age"],
        row["Sex-male"],
        row["Sex-female"],
        row.get("Cabin", {}).get("deck") == "A",
        row.get("Cabin", {}).get("deck") == "B",
        row.get("Cabin", {}).get("deck") == "C",
        row.get("Cabin", {}).get("deck") == "D",
        row.get("Cabin", {}).get("deck") == "E",
        row.get("Cabin", {}).get("deck") == "F",
        row.get("Cabin", {}).get("deck") == "G",
        # don't use cabin number.. for now
        row["SibSp"],
        row["Parch"],
        row["Fare"],
        row["Embarked-Cherbourg"],
        row["Embarked-Queenstown"],
        row["Embarked-Southampton"]
    ], dtype=np.float32)
    return output


def row_to_model_label(row):
    return np.float32([row["Survived"]])


def read_train_data():
    with open("data/train.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = []
        for row in csv_reader:
            row = transform_row(row)
            rows.append(row)
        return rows


def train():
    train_data = read_train_data()
    dataset = [[row_to_model_input(row), row_to_model_label(row)] for row in train_data]
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model = torch.nn.Sequential(
        torch.nn.Linear(len(dataset[0][0]), 16),
        torch.nn.Linear(16, 16),
        torch.nn.Linear(16, 16),
        torch.nn.Linear(16, 8),
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 1),
        torch.nn.Sigmoid()
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(100):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # print statistics
        print('[epoch %d] loss: %.3f' %
              (epoch + 1, running_loss / len(train_data)))
    print('finished training')

    with torch.no_grad():
        all_inputs = torch.FloatTensor([data[0] for data in dataset])
        all_labels = torch.FloatTensor([data[1] for data in dataset])
        all_outputs = model(all_inputs)
        loss = loss_fn(all_outputs, all_labels)
        print('total train loss: %.3f' % (loss.item() / len(all_inputs)))

    return model


def read_test_data():
    with open("data/test.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = []
        for row in csv_reader:
            row = transform_row(row)
            rows.append(row)
        return rows


def run_test_extrapolation(pytorch_model):
    test_data = read_test_data()
    test_inputs = torch.FloatTensor([row_to_model_input(row) for row in test_data])
    outputs = pytorch_model(test_inputs) > 0.5
    os.remove('test_output.csv')
    with open('test_output.csv', 'w') as csvfile:
        fieldnames = ['PassengerId', 'Survived']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(test_inputs)):
            writer.writerow({'PassengerId': test_data[idx]["PassengerId"],
                             'Survived': 1 if outputs[idx].item() else 0})


if __name__ == "__main__":
    # data = read_train_data()
    # decks = [row['Cabin']['deck'] for row in data if 'Cabin' in row]
    # cabin = [ord(deck) - ord('A') for deck in decks]
    # survived = [row["Survived"] for row in data if 'Cabin' in row]
    # plt.hist2d(cabin, survived, bins=[8, 2])
    # plt.xlabel("Cabin")
    # plt.ylabel("Survived")
    # plt.show()
    model = train()
    run_test_extrapolation(model)

