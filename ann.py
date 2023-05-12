import math
import pickle
import torch
import torch.nn as nn

input_size = 7
hidden_size = 21
num_classes = 3
num_epochs = 100
batch_size = 10000
learning_rate = 0.01


class NeuralNetwork(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_layer, hidden_layer)
        self.activation1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer, hidden_layer)
        self.activation2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_layer, output_layer)
        self.activation3 = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.l1(x)
        out = self.activation1(out)
        out = self.l2(out)
        out = self.activation2(out)
        out = self.l3(out)
        out = self.activation3(out)
        return out


def Prediction(output):
    prediction = []
    for i in range(len(output)):
        c = torch.argmax(output[i])
        if c == 0:
            prediction.append([1, 0, 0])
        elif c == 1:
            prediction.append([0, 1, 0])
        else:
            prediction.append([0, 0, 1])
    return prediction


if __name__ == "__main__":
    device = torch.device('cpu')

    with open("ann_input.pkl", "rb") as file:
        input_data = pickle.load(file)
        file.close()

    with open("labels.pkl", "rb") as file:
        labels = pickle.load(file)
        file.close()

    input_data = torch.tensor(input_data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    train_set = input_data[:50000]
    train_labels = labels[:50000]
    records = len(train_labels)
    num_batches = math.floor(records / batch_size)

    model = NeuralNetwork(input_size, hidden_size, num_classes)

    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i in range(num_batches):
            input_batch = train_set[i * batch_size:(i + 1) * batch_size]
            label_batch = train_labels[i * batch_size:(i + 1) * batch_size]
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)

            outputs = model(input_batch)
            loss = criteria(outputs, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_set = input_data[50001:90000]
    test_labels = labels[50001:90000]
    predictions = Prediction(model(test_set))
    predictions = torch.tensor(predictions, dtype=torch.float32)
    true_predictions = 0
    for i in range(len(test_labels)):
        if torch.equal(test_labels[i], predictions[i]):
            true_predictions += 1
    accuracy = (true_predictions / len(test_labels)) * 100
    print(true_predictions)
    print(accuracy)
