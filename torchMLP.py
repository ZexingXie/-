from torch.autograd import Variable
from torch import optim
from torch import nn
import torch as t
import numpy as np
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from util import data_loads
from util import evaluate


# 二分类
class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        # input:size is M*2. M　is the batch　number
        # target:size is M.
        pt = t.softmax(input, dim=1)
        p = pt[:, 1]
        target = target.float()
        # print(p)
        loss = -self.alpha * (1 - p) ** self.gamma * (target * t.log(p)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * t.log(1 - p))
        return loss.mean()




class TorchDemo(nn.Module):
    def __init__(self):
        super(TorchDemo, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100,40),
            nn.SELU(),
            nn.Linear(40, 20),
            nn.SELU(),
            nn.Linear(20,2),
            nn.LogSoftmax(dim=1),
        )
        self.declare_parameters()
    def declare_parameters(self):
        self.build_loss_function()
        self.build_optimizer()
    def build_loss_function(self):
        # self.criterion = nn.NLLLoss()
        pass
    def build_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-6)
    def forward(self,x):
        out = self.model(x)
        return out
    def compute_cost(self, out, targes):
        self.criterion = FocalLoss()
        return self.criterion.forward(out,targes)
    def optimize(self, inputs, targes):
        self.optimizer.zero_grad()
        out = self.forward(inputs)
        self.loss = self.compute_cost(out, targes)
        self.loss.backward()
        self.optimizer.step()
        return self.loss
    def train(self, x_train, y_train, num_epochs=60000):
        if isinstance(x_train,np.ndarray):
            x_train = t.from_numpy(x_train).float()
            y_train = t.from_numpy(y_train).long()
        inputs = Variable(x_train)
        targes = Variable(y_train)
        self.num_epochs = num_epochs
        for epoch in range(self.num_epochs):
            loss = self.optimize(inputs, targes)
            if (epoch + 1) % 20 == 0:
                print('Epoch[{}/{}],loss:{:.6f}'.format(
                    epoch + 1, num_epochs, loss.data.numpy()))
                t.save(self.model.state_dict(), './torchsave.pt')

    def predict(self, x_test):
        self.model.load_state_dict(t.load('./torchsave.pt'))
        if isinstance(x_test, np.ndarray):
            x_test = t.from_numpy(x_test).float()
        predict_y = self.forward(x_test).data.numpy()
        return  [np.argmax(one_hot)for one_hot in predict_y]

if __name__=='__main__':
    loads = data_loads.Data_Load()
    batch_xs, batch_ys = loads.chooceall()
    dnn = TorchDemo()
    dnn.train(batch_xs,batch_ys)
    predict_y = dnn.predict(batch_xs)
    evaluate.evaluate(batch_ys, predict_y)

    batch_xs, batch_ys = loads.all()
    predict_y = dnn.predict(batch_xs)
    evaluate.evaluate(batch_ys, predict_y)
