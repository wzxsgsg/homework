import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange
import json
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

data = []
for i in range(1, 101):
    filePath = 'D:\\pythonProject\\test\\data\\'
    with open(filePath+str(i)+'.json') as f:
        line = f.readline()
        d = json.loads(line)
        x_data = d["x"]
        y_label = d["label"]
        data.append([x_data, y_label])
        f.close()

train_data = []
test_data = []

for i in range(len(data)):
    if i < 80:
        train_data.append(data[i])
    else:
        test_data.append(data[i])

# print(train_data)
# print(test_data)

def create_batches(data):
    #random.shuffle(data)
    batches = [data[graph:graph+16] for graph in range(0, len(data), 16)]
    return batches

from nnModel import NeuralNet

model = NeuralNet(48, 500, 150, 2)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion=nn.MSELoss()

min_loss = None
epochs = trange(10, leave=True, desc = "Epoch")
for epoch in epochs:
    batches = create_batches(train_data)
    totalloss = 0.0
    main_index = 0.0
    for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
        batchloss = 0
        optimizer.zero_grad()
        for data, label in batch:
            data = torch.tensor(data, device=device)
            label = torch.tensor(label, device=device)
            output = model(data)
            loss = criterion(output, label)
            batchloss = batchloss+loss
        batchloss.backward()
        optimizer.step()
        loss = batchloss.item()
        totalloss += loss
        main_index = main_index + len(batch)
        loss = totalloss / main_index
        if min_loss is None or min_loss > loss:
            min_loss = loss
            torch.save(model.state_dict(), 'modelsave/bestmodel')
        epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

def F(x, y):
    return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)

def test(model, dataset):
    correct = 0
    for data in dataset:
        x = data[0]
        y = data[1]
        x = torch.tensor(x, device=device)
        y = torch.tensor(y, device=device)
        y_pre = model(x)
        y_pre_0 = y_pre[0].item()
        y_pre_1 = y_pre[1].item()
        # y_pre_0 = format(y_pre_0, '.3f')
        # y_pre_1 = format(y_pre_1, '.3f')
        y_0 = y[0].item()
        y_1 = y[1].item()
        # y_0 = format(y_0, '.3f')
        # y_1 = format(y_1, '.3f')
        print("y_pre_0:{}, y_0:{}, y_pre_1:{}, y_1:{}".format(y_pre_0, y_0, y_pre_1, y_1))
        if F(y_pre_0, y_pre_1) > F(y_0, y_1):
            correct = correct+1
    acc = correct/len(dataset)
    return acc

model.load_state_dict(torch.load('modelsave/bestmodel'))
acc = test(model, test_data)
print("{:.2f}".format(acc))