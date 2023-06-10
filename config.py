import torch
import sys
sys.path.append("/home/ljq/argoverse-api/")

if torch.cuda.is_available():
#if 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model_save_path = "model/"

def get_ADE(pred, target):
    #target = target.to(config['device'])
    assert pred.shape == target.shape
    temp=torch.sum((pred - target) ** 2)
    tmp = torch.sqrt(torch.sum((pred - target) ** 2, dim=2)) # [batch_size, len]
    ade = torch.mean(tmp, dim=1, keepdim=True) # [batch_size, 1]
    return ade 
def get_FDE(pred, target):
    #target = target.to(config['device'])
    assert pred.shape == target.shape  #batch_size, 30, 2
    pred = pred[:, -1, :] # [batch_size, dim]
    target = target[:, -1, :]
    fde = torch.sqrt(torch.sum((pred - target) ** 2, dim=1, keepdim=True)) # [batch_size, 1]
    return fde
def get_1DE(pred, target):
    #target = target.to(config['device'])
    assert pred.shape == target.shape  #batch_size, 30, 2
    pred = pred[:, 9, :] # [batch_size, dim]
    target = target[:, 9, :]
    fde = torch.sqrt(torch.sum((pred - target) ** 2, dim=1, keepdim=True)) # [batch_size, 1]
    return fde  
def get_2DE(pred, target):
    #target = target.to(config['device'])
    assert pred.shape == target.shape  #batch_size, 30, 2
    pred = pred[:, 19, :] # [batch_size, dim]
    target = target[:, 19, :]
    fde = torch.sqrt(torch.sum((pred - target) ** 2, dim=1, keepdim=True)) # [batch_size, 1]
    return fde  

