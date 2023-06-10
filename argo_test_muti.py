import torch
import sys
from config import get_ADE,get_FDE,get_1DE,get_2DE
import config
from network.vector_net import VectorNet, VectorNetWithPredicting
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def get_ADE(pred, target):
    #target = target.to(config['device'])
    assert pred.shape == target.shape
    temp=torch.sum((pred - target) ** 2)
    tmp = torch.sqrt(torch.sum((pred - target) ** 2, dim=2)) # [batch_size, len]
    ade = torch.mean(tmp, dim=1, keepdim=True) # [batch_size, 1]
    return ade 

dataset20 = np.load("data_test_1000.npy",
                   allow_pickle=True)

dataset = dataset20
data_len = len(dataset)
n = int(0*len(dataset))
print(data_len)
dataset = dataset[n:data_len-1]
batch = 1
train_loader = DataLoader(dataset, batch_size=batch, shuffle=True,drop_last=True)


device = torch.device("cuda")
print("load")
vector_net = torch.load("/home/ljq/motion_prediction/pre_train")
vector_net = vector_net.to(device)
optimizer = torch.optim.Adam(vector_net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()
vector_net.eval()
cnt=0
all_loss=0
all_loss1=0
all_loss2=0
all_loss3=0
all_loss4=0
print("start test")
for i, data in enumerate(train_loader):
    #print(i)
    labels = data["gt_preds"].to(torch.float32)
    #print(labels.shape)
    idd = torch.tensor([0]*batch)
    outputs = vector_net(data["item_num"].to(config.device), idd.to(
            config.device), data["polyline_list1"], data['near_centerline'].to(
            config.device), data['goal'].to(
            config.device),0)
    #print(outputs['traj_with_gt'])
    #print(outputs['traj_with_gt'].shape,labels.shape
    feature = data["polyline_list1"]
    outputs1 = outputs['traj_with_gt'].squeeze().detach().cpu().numpy()
    outputs2 = outputs['trajs'].squeeze().detach().cpu().numpy()
    #target_points=outputs['target_select'].squeeze().detach().cpu().numpy()
    #print("test: ",outputs2.shape,outputs1.shape)

    labels1 = labels.squeeze().detach().cpu().numpy()
    loss = loss_func(outputs['traj_with_gt'], labels.to(config.device))
    # print(loss.shape)
    ade = torch.mean(get_ADE(outputs['traj_with_gt'], labels.to(
        config.device))).cpu().detach().numpy()
    fde = torch.mean(get_FDE(outputs['traj_with_gt'], labels.to(
        config.device))).cpu().detach().numpy()

    onede = torch.mean(get_1DE(outputs['traj_with_gt'], labels.to(
        config.device))).cpu().detach().numpy()
    twode = torch.mean(get_2DE(outputs['traj_with_gt'], labels.to(
        config.device))).cpu().detach().numpy()


    all_loss = all_loss+float(ade)
    all_loss1 = all_loss1+float(loss.item()/batch)
    all_loss2 = all_loss2+float(fde)
    all_loss3 = all_loss3+float(onede)
    all_loss4 = all_loss4+float(twode)

    all_traj=outputs['trajs'].squeeze().detach().cpu().numpy()
    
    near_cente=data['near_centerline']
 
    plt.cla()
    plt.axis('equal')   
    plt.scatter(near_cente[0,:, 0], near_cente[0, :, 1], color="black", linewidth=1)
    #print("dd: ",target_points.shape)
    #plt.scatter(target_points[:, 0], target_points[:, 1], color="red", linewidth=2)
    
    # for k in range(5):
    #     each_traj = all_traj[k, :]
    #     plt.scatter(each_traj[:, 0], each_traj[:, 1], s=5, c='c')
    plt.plot(labels1[:, 0], labels1[:, 1], color="black", linewidth=5)
    plt.plot(outputs1[:, 0], outputs1[:, 1], color="green", linewidth=5)
    colorset = ["red","yellow","pink","magenta","bisque"]
    for i in range(5):
        #print(outputs2[i])
        plt.plot(outputs2[i,:, 0], outputs2[i,:, 1], color=colorset[i], linewidth=1)
    for i in range(30):
        plt.scatter(feature[i][0, :, 0], feature[i][0, :, 1], s=5)
    for j in range(90):
        plt.scatter(feature[30+j][0, :, 0], feature[30+j][0, :, 1], s=5, c='r')
    plt.show()

    cnt=cnt+1
    
print("ade: ",cnt, all_loss/(cnt), "fde: ",all_loss2/(cnt),all_loss3/(cnt),all_loss4/(cnt))
