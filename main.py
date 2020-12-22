from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

def read_fold():
    allTrain = None
    allTest = None
    with open('./kfold_train.bin', 'rb') as f:
        allTrain = pickle.load(f)
    with open('./kfold_test.bin', 'rb') as f:
        allTest = pickle.load(f)
    return allTrain, allTest

def svr_helper():
    raw = None
    with open('./data.bin', 'rb') as f:
        raw = pickle.load(f)
    allTrain, allTest = read_fold()
    for i in range(10):
        print("Fold {}".format(i + 1))
        train_features = raw[allTrain[i], :-3]
        train_alpha = raw[allTrain[i], -3]
        train_t1 = raw[allTrain[i], -2]
        train_t2 = raw[allTrain[i], -1]

        test_features = raw[allTest[i], :-3]
        test_alpha = raw[allTest[i], -3]
        test_t1 = raw[allTest[i], -2]
        test_t2 = raw[allTest[i], -1]

        # 预测alpha
        model1 = SVR(gamma='auto')
        model1.fit(train_features, train_alpha)
        print(model1.score(test_features, test_alpha))

        # 预测t1
        model2 = SVR(gamma='auto')
        model2.fit(train_features, train_t1)
        print(model2.score(test_features, test_t1))

        # 预测t2
        model3 = SVR(gamma='auto')
        model3.fit(train_features, train_t2)
        print(model3.score(test_features, test_t2))

def rfr_helper():
    raw = None
    with open('./data.bin', 'rb') as f:
        raw = pickle.load(f)
    allTrain, allTest = read_fold()
    for i in range(10):
        print("Fold {}".format(i + 1))
        train_features = raw[allTrain[i], :-3]
        train_alpha = raw[allTrain[i], -3]
        train_t1 = raw[allTrain[i], -2]
        train_t2 = raw[allTrain[i], -1]

        test_features = raw[allTest[i], :-3]
        test_alpha = raw[allTest[i], -3]
        test_t1 = raw[allTest[i], -2]
        test_t2 = raw[allTest[i], -1]

        # 预测alpha
        model1 = RandomForestRegressor(n_estimators=100)
        model1.fit(train_features, train_alpha)
        print(model1.score(test_features, test_alpha))

        # 预测t1
        model2 = RandomForestRegressor(n_estimators=100)
        model2.fit(train_features, train_t1)
        print(model2.score(test_features, test_t1))

        # 预测t2
        model3 = RandomForestRegressor(n_estimators=100)
        model3.fit(train_features, train_t2)
        print(model3.score(test_features, test_t2))


class scaled_dot_product_attention(nn.Module):
    def __init__(self, att_dropout=0.0):
        super(scaled_dot_product_attention, self).__init__()
        self.dropout = nn.Dropout(att_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None):
        '''
        args:
            q: [batch_size, q_length, q_dimension]
            k: [batch_size, k_length, k_dimension]
            v: [batch_size, v_length, v_dimension]
            q_dimension = k_dimension = v_dimension
            scale: 缩放因子
        return:
            context, attention
        '''
        # 快使用神奇的爱因斯坦求和约定吧！
        attention = torch.einsum('ijk,ilk->ijl', [q, k])
        if scale:
            attention = attention * scale
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.einsum('ijl,ilk->ijk', [attention, v])
        return context, attention

class multi_heads_self_attention(nn.Module):
    def __init__(self, feature_dim, num_heads=1, dropout=0.0):
        super(multi_heads_self_attention, self).__init__()

        self.dim_per_head = feature_dim // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(feature_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(feature_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(feature_dim, self.dim_per_head * num_heads)

        self.sdp_attention = scaled_dot_product_attention(dropout)
        self.linear_attention = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        # self.linear_1 = nn.Linear(feature_dim, 256)
        # self.linear_2 = nn.Linear(256, feature_dim)
        # self.layer_final = nn.Linear(feature_dim, 3)

    def forward(self, key, value, query):
        residual = query
        batch_size = 1

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if key.size(-1) // self.num_heads != 0:
            scale = (key.size(-1) // self.num_heads) ** -0.5
        else:
            scale = 1
        context, attention = self.sdp_attention(query, key, value, scale)

        # concat heads
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)

        output = self.linear_attention(context)
        output = self.dropout(output)
        # output = torch.squeeze(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        # # pass through linear
        # output = nn.functional.relu(self.linear_1(output))
        # output = nn.functional.relu(self.linear_2(output))

        # # pass through layer final
        # output = self.layer_final(output)

        return output, attention

class cross_stitch(nn.Module):
    def __init__(self, length):
        '''
        length是两个input的最后一个维度和
        '''
        super(cross_stitch, self).__init__()
        self.matrix = nn.Parameter(torch.empty(length, length))
        nn.init.uniform_(self.matrix, 0.5, 0.8)

    def forward(self, input1, input2):
        # 这里数据除开batch，本来就是1维，不需要展平了
        input1_reshaped = torch.squeeze(input1, dim=0)
        input2_reshaped = torch.squeeze(input2, dim=0)
        input_reshaped = torch.cat([input1_reshaped, input2_reshaped], dim=-1)
        output = torch.matmul(input_reshaped, self.matrix)

        output1 = torch.reshape(output[:, :input1.size()[-1]], input1.size())
        output2 = torch.reshape(output[:, input2.size()[-1]:], input2.size())

        return output1, output2

class MultiLossLayer(nn.Module):
    def __init__(self, list_length):
        super(MultiLossLayer, self).__init__()
        self._sigmas_sq = nn.ParameterList([nn.Parameter(torch.empty(())) for i in range(list_length)])
        for p in self.parameters():
            nn.init.uniform_(p, 0.1, 0.2)
        
    def forward(self, loss0, loss1, loss2):
        # loss0
        factor0 = torch.div(1.0, torch.mul(self._sigmas_sq[0], 2.0))
        loss = torch.add(torch.mul(factor0, loss0), 0.5 * torch.log(self._sigmas_sq[0]))
        # loss1
        factor1 = torch.div(1.0, torch.mul(self._sigmas_sq[1], 2.0))
        loss = torch.add(loss, torch.add(torch.mul(factor1, loss1), 0.5 * torch.log(self._sigmas_sq[1])))
        # loss2
        factor2 = torch.div(1.0, torch.mul(self._sigmas_sq[2], 2.0))
        loss = torch.add(loss, torch.add(torch.mul(factor2, loss2), 0.5 * torch.log(self._sigmas_sq[2])))

        return loss

class CrossModel(nn.Module):
    def __init__(self, length):
        super(CrossModel, self).__init__()
        # 因为有三个任务，所以需要有三路模型，中间两个十字绣单元
        # 每一路都采用两个attention模块，最后接一个线形层转化为最后的目标值
        self.attention1_1 = multi_heads_self_attention(feature_dim=length)
        self.attention1_2 = multi_heads_self_attention(feature_dim=length)
        self.linear1 = nn.Linear(length, 1)

        self.attention2_1 = multi_heads_self_attention(feature_dim=length)
        self.attention2_2 = multi_heads_self_attention(feature_dim=length)
        self.linear2 = nn.Linear(length, 1)

        self.attention3_1 = multi_heads_self_attention(feature_dim=length)
        self.attention3_2 = multi_heads_self_attention(feature_dim=length)
        self.linear3 = nn.Linear(length, 1)

        self.cross_stitch1 = cross_stitch(length*2)
        self.cross_stitch2 = cross_stitch(length*2)

        self.loss1 = nn.MSELoss()
        self.loss2 = nn.MSELoss()
        self.loss3 = nn.MSELoss()

        self.lossLayer = MultiLossLayer(list_length=3)

    def forward(self, input, y1, y2, y3):
        # y1=fatigue; y2=tensile; y3=fracture; y4=hardness;
        # 统一描述：out_路编号_阶段编号

        # 第一阶段
        out1_1, _ = self.attention1_1(input, input, input)
        out2_1, _ = self.attention2_1(input, input, input)
        out3_1, _ = self.attention3_1(input, input, input)
        out1_1, out2_1_a = self.cross_stitch1(out1_1, out2_1)
        out2_1_b, out3_1 = self.cross_stitch2(out2_1, out3_1)
        out2_1 = 0.5 * out2_1_a + 0.5 * out2_1_b
        # 第二阶段
        out1_2, _ = self.attention1_2(out1_1, out1_1, out1_1)
        out2_2, _ = self.attention2_2(out2_1, out2_1, out2_1)
        out3_2, _ = self.attention3_2(out3_1, out3_1, out3_1)
        # 第三阶段
        out1_3 = nn.functional.relu(self.linear1(out1_2))
        out2_3 = nn.functional.relu(self.linear2(out2_2))
        out3_3 = nn.functional.relu(self.linear3(out3_2))

        # 分任务计算损失
        loss1 = self.loss1(torch.squeeze(out1_3), torch.squeeze(y1))
        loss2 = self.loss2(torch.squeeze(out2_3), torch.squeeze(y2))
        loss3 = self.loss3(torch.squeeze(out3_3), torch.squeeze(y3))

        loss = self.lossLayer(loss1, loss2, loss3)

        return loss, out1_3, out2_3, out3_3

def mtl_helper():
    raw = None
    with open('./data.bin', 'rb') as f:
        raw = pickle.load(f)
    allTrain, allTest = read_fold()

    i = 9

    x_train = raw[allTrain[i], :-3]
    y_train = raw[allTrain[i], -3:]

    x_test = raw[allTest[i], :-3]
    y_test = raw[allTest[i], -3:]

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    # x_train, y_train = train.values[:, :-3], train.values[:, -3:]
    # x_test, y_test = test.values[:, :-3], test.values[:, -3:]

    if torch.cuda.is_available():
        x_train = torch.from_numpy(x_train).to(device)
        x_test = torch.from_numpy(x_test).to(device)

        alpha_train = torch.from_numpy(y_train[:, 2]).view(-1, 1).to(device)
        alpha_test = torch.from_numpy(y_test[:, 2]).view(-1, 1).to(device)

        t1_train = torch.from_numpy(y_train[:, 0]).view(-1, 1).to(device)
        t1_test = torch.from_numpy(y_test[:, 0]).view(-1, 1).to(device)

        t2_train = torch.from_numpy(y_train[:, 1]).view(-1, 1).to(device)
        t2_test = torch.from_numpy(y_test[:, 1]).view(-1, 1).to(device)
    else:
        x_train = torch.from_numpy(x_train)
        x_test = torch.from_numpy(x_test)

        alpha_train = torch.from_numpy(y_train[:, 2]).view(-1, 1)
        alpha_test = torch.from_numpy(y_test[:, 2]).view(-1, 1)

        t1_train = torch.from_numpy(y_train[:, 0]).view(-1, 1)
        t1_test = torch.from_numpy(y_test[:, 0]).view(-1, 1)

        t2_train = torch.from_numpy(y_train[:, 1]).view(-1, 1)
        t2_test = torch.from_numpy(y_test[:, 1]).view(-1, 1)

    model = CrossModel(length=x_train.size()[1])
    if torch.cuda.is_available():
        model.to(device)
    model.double()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch_num = 10000

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            loss, _, _, _ = model(x_train, alpha_train, t1_train, t2_train)
            loss.backward()

            return loss

        optimizer.step(closure)

        if epoch % 10 == 9:
            print('epoch: {}'.format(epoch))

            loss, pred1, pred2, pred3 = model(x_test, alpha_test, t1_test, t2_test)
            _, out1, out2, out3 = model(x_train, alpha_train, t1_train, t2_train)
            print('loss: {}'.format(loss))

            r21 = r2_score(torch.squeeze(pred1.cpu()).detach().numpy(), torch.squeeze(alpha_test.cpu()).detach().numpy())
            r22 = r2_score(torch.squeeze(pred2.cpu()).detach().numpy(), torch.squeeze(t1_test.cpu()).detach().numpy())
            r23 = r2_score(torch.squeeze(pred3.cpu()).detach().numpy(), torch.squeeze(t2_test.cpu()).detach().numpy())
            r2 = r21 + r22 + r23
            print('r2_1: {}\nr2_2: {}\nr2_3: {}\nr2: {}'.format(r21, r22, r23, r2))


if __name__ == '__main__':
    # read_fold()
    # foldData()
    svr_helper()
    # rfr_helper()
    # mtl_helper()