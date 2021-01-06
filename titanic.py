#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import numpy as np
import re
import torch


class MixtureOfLogistics(torch.nn.Module):
    def __init__(self, cardinality, num_logistics):
        super(MixtureOfLogistics, self).__init__()
        self.cardinality = cardinality
        self.num_logistics = num_logistics
        self.mu = torch.nn.Parameter(torch.arange(0,
                                                  self.cardinality,
                                                  self.cardinality/self.num_logistics,
                                                  dtype=torch.float))
        self.pi = torch.nn.Parameter(torch.ones(num_logistics,
                                                dtype=torch.float) / num_logistics)
        self.s = torch.nn.Parameter(torch.ones(num_logistics,
                                               dtype=torch.float))

    def forward(self, x):
        x = torch.Tensor(x)
        # num_values x num_logistics
        x = x.unsqueeze(1).repeat(1, self.num_logistics)
        upper_cdf = torch.where(x >= self.cardinality - 1,
                                torch.ones(x.size()),
                                torch.sigmoid((x + 0.5 - self.mu)/self.s))
        lower_cdf = torch.where(x == 0,
                                torch.zeros(x.size()),
                                torch.sigmoid((x - 0.5 - self.mu)/self.s))
        summed_sigmoids = upper_cdf - lower_cdf

        pi_softmax = torch.nn.functional.softmax(self.pi, dim=0)
        weighted_sum = pi_softmax * summed_sigmoids
        final_probability = torch.sum(weighted_sum, dim=1)
        return final_probability

    def loss(self, x):
        return -torch.mean(torch.log(self(x)))


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
    if row["Survived"]:
        row["Survived"] = int(row["Survived"])

    # 1, 2, or 3
    row["Pclass"] = int(row["Pclass"])

    # Sex can be "male" or "female"

    if row["Age"]:
        row["Age"] = float(row["Age"])

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
    row["Fare"] = float(row["Fare"])

    row["Embarked-Cherbourg"] = row["Embarked"] == "C"
    row["Embarked-Queenstown"] = row["Embarked"] == "Q"
    row["Embarked-Southampton"] = row["Embarked"] == "S"
    return row


def row_to_model_input(row):
    output = np.array([
        row["Pclass"] == 1,
        row["Pclass"] == 2,
        row["Pclass"] == 3,
        row["Age"] or 35,
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
        torch.nn.Linear(len(dataset[0][0]), 20),
        torch.nn.Linear(20, 10),
        torch.nn.Linear(10, 1),
        torch.nn.Sigmoid()
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    running_loss = 0.0
    for epoch in range(10):
        for i, (inputs, labels) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    print('finished training')

    with torch.no_grad():
        all_inputs = torch.FloatTensor([data[0] for data in dataset])
        all_labels = torch.FloatTensor([data[1] for data in dataset])
        all_outputs = model(all_inputs)
        loss = loss_fn(all_outputs, all_labels)
        print('total train loss: %.3f' % (loss.item() / len(all_inputs)))

    return model


def read_test_data():
    pass


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

