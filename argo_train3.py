import torch
import sys
import config
from config import get_ADE,get_FDE,get_1DE,get_2DE
from network.vector_net import VectorNet, VectorNetWithPredicting
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import math
from pathlib import Path
import torch.nn.functional as F
def distance_metric(traj_candidate: torch.Tensor, traj_gt: torch.Tensor):

    _, M, horizon_2_times,_ = traj_candidate.size()
    dis = torch.pow(traj_candidate - traj_gt.unsqueeze(1), 2).view(-1, M, int(horizon_2_times), 2)
    #print("ds: ",dis.shape)
    dis, _ = torch.max(torch.sum(dis, dim=3), dim=2)
    return dis

def calc_loss(output, label):
    traj_gt = outputs['traj_with_gt']
    trajs = outputs['trajs']
    score = outputs['score']
    mask_target_p_select = outputs['mask_target_p_select']
    target_select = outputs['target_select']
    offset = outputs['offset']
    #print("debug1: ",target.shape)
    loss = 0
   # print("loss: ",label.shape,trajs.shape,traj_gt.shape,mask_target_p_select.shape,target_select.shape)
   #print("loss1 shape: ", traj_gt.shape, trajs.shape)
    loss1 = F.mse_loss(traj_gt, label)
    score_gt = F.softmax(-distance_metric(trajs, traj_gt)/10.0, dim=1)
    #print("loss2 shape: ", score_gt[0,:])

    logprobs = - torch.log(score)
    loss2 = torch.sum(torch.mul(logprobs, score_gt))

    loss3 = F.binary_cross_entropy(
        target_select.float(), mask_target_p_select.float(), reduction='none').sum()
    t_offset =torch.zeros_like(offset)
    loss4 = F.mse_loss(t_offset, offset)
    print(loss1.item(),loss2.item(),loss3.item(),loss4.item())
    loss =loss1+loss2*0.01+loss3*0.01+loss4*10.0
    return loss

dataset0 = np.load("data_train_5000.npy",
                   allow_pickle=True)
print("load 1")
dataset_train = dataset0

batch = 64
torch.multiprocessing.set_sharing_strategy('file_system')
#train_loader = DataLoader(dataset, batch_size=batch, drop_last=True)
train_loader = DataLoader(
    dataset_train,
    batch_size=batch,
    shuffle=True,
    drop_last=True
)

# val_loader = DataLoader(
#     dataset_val,
#     batch_size=batch,
#     shuffle=False,
#     drop_last=True
# )
print("start train")

vector_net = VectorNetWithPredicting(12, time_stamp_number=30)
#vector_net = torch.load("/home/ljq/argoverse-api/ljq_test/net/3")
vector_net = vector_net.to(config.device)
learning_rate=0.001
decayed_factor=0.3
optimizer = torch.optim.Adam(vector_net.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()

loss_max=9999999999999
loss_max1=9999999999999
cnt = 0
vector_net.train()
for epoch in range(200):
    train_loss = 0
    val_loss=0

    cnt_train=1
    cnt_val=1
    for i, data in enumerate(train_loader):
        #print(i,data)
        cnt_train=i+1
        optimizer.zero_grad()
        labels = data["gt_preds"].to(torch.float32)
        # print(len(data["polyline_list1"]),
        #       data["item_num"].shape, data["idx"].shape)
        idd = torch.tensor([0]*batch)   
        outputs = vector_net(data["item_num"].to(config.device), idd.to(
            config.device), data["polyline_list1"], data['near_centerline'].to(
            config.device), data['goal'].to(
            config.device),epoch)

        # loss = loss_func(outputs['traj_with_gt'], labels.to(config.device))
        loss = calc_loss(outputs,labels.to(config.device))
        
        loss.backward()
        optimizer.step()
        train_loss = train_loss+loss.item()

    if train_loss<loss_max:
        diff=math.fabs(train_loss-loss_max)/loss_max
        loss_max=train_loss
        if epoch %10==0:
            torch.save(vector_net, "/home/ljq/motion_prediction/save/"+str(epoch)+"  "+str(loss_max))
    #     print("sub ",diff)
    #     if  diff<0.05:
    #         print("sub1 ",diff)
    #         learning_rate *= decayed_factor
    #         optimizer = torch.optim.Adam(vector_net.parameters(), lr=learning_rate)
    if epoch != 0 and epoch % 20 == 0:
        learning_rate *= decayed_factor
        learning_rate =max(learning_rate,0.0003)
        optimizer = torch.optim.Adam(vector_net.parameters(), lr=learning_rate)
    cnt = cnt+1
    
    train_loss=train_loss/cnt_train/batch
    print("epoch  ",epoch," train loss: ",train_loss,learning_rate)
