
import torch
import numpy as np
import sys

def getData(xPath):
  x_in = np.genfromtxt(xPath ,delimiter=",")
  x = x_in[:,1:]
  y = x_in[:,0]
  # x = x/255
  return x,y


xTrain,yTrain = getData(str(sys.argv[1]))
xTest,yTest = getData(str(sys.argv[2]))
r = len(np.unique(yTrain))
n = np.shape(xTrain)[1]
numFeatures = n
m = np.shape(xTrain)[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.nn.Sequential(
        torch.nn.Linear(in_features=n, out_features=100),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=100, out_features=r)
    )

# print(model)

# model = model.cuda() ##Check whether cuda is available or not
xTrain = torch.tensor(xTrain,dtype=torch.float).to(device)
yTrain = torch.tensor(yTrain,dtype=torch.long).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # stochastic gradient descent
criterion = torch.nn.CrossEntropyLoss()

lossList = []
epochList = []

EPOCHS = 2000
M = 100
model = model.to(device)
c = 0
prevJ = 0
earlyStop = 0
epoch = 0
while epoch<100 and earlyStop<10:
    epoch =0
    for i in range(0,int(m/M)):
      
      # optimizer.zero_grad()
      x_train = (xTrain[(i)*M:(i)*M + M,:]).to(device)
      y_train = (yTrain[i*M:i*M + M]).to(device)
      outputs = model(x_train)
      loss = criterion(outputs, y_train)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()      
      c+=1
      if (i+1) % 50 == 0:
        lossList.append(loss.item())
        epochList.append(c)
        # print(abs(loss.item()-prevJ))
        # print ('Loss: {:.4f}'.format(loss.item()))
      if abs(loss.item()-prevJ) < 1e-5:
        earlyStop += 10
      prevJ = loss.item()

# y_predicted = model(xTrain)
# # print(y_predicted)
# y_predicted = torch.argmax(y_predicted,dim=1)
# # print(y_predicted)
# # print(yTrain)
# # print(yTrain.size())
# # yTrain = yTrain.to(device)
# correct = 0
# for i in range(len(y_predicted)):
#   if yTrain[i] == y_predicted[i]:
#     correct += 1
# # yTrain = yTrain.view(m,-1)
# # correct += (y_predicted == yTrain).float().sum()
# # print(correct)
# accuracy = 100 * correct / ((xTrain.size)(0))
# print("Accuracy = {}".format(accuracy))

xTest = torch.tensor(xTest,dtype=torch.float).to(device)
yTest = torch.tensor(yTest,dtype=torch.long).to(device)
y_predicted = model(xTest)
# print(y_predicted)
y_predicted = torch.argmax(y_predicted,dim=1)

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")
# np.savetxt(sys.argv[3], y_predicted, fmt="%d", delimiter="\n")
write_predictions(str(sys.argv[3]),y_predicted.cpu())
# print(y_predicted)
# print(yTrain)
# print(yTrain.size())
# correct = 0
# for i in range(len(y_predicted)):
#   if yTest[i] == y_predicted[i]:
#     correct += 1
# # yTrain = yTrain.view(m,-1)
# # correct += (y_predicted == yTrain).float().sum()
# # print(correct)
# accuracy = 100 * correct / ((xTest.size)(0))
# print("Accuracy = {}".format(accuracy))

# plt.plot(epochList,lossList)

# confusion_matrix = torch.zeros(7, 7)
# y_predicted = model(xTest)
# print(y_predicted)
# y_predicted = torch.argmax(y_predicted,dim=1)
# # print(y_predicted)
# # print(yTrain)
# # print(yTrain.size())
# correct = 0
# for t, p in zip(yTest.view(-1), y_predicted.view(-1)):
#   confusion_matrix[t.long(), p.long()] += 1

# true_positive = torch.eq(yTest, y_predicted).sum().float()
# f1_score = torch.div(true_positive, len(yTest))

# print(confusion_matrix.int())

# print(f1_score)

# from torchsummary import summary
# summary(model, (2304,))

