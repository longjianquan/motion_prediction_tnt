
import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden=64, bias=True, activation="relu", norm='layer'):
        super(MLP, self).__init__()
        act_layer = nn.ReLU
        norm_layer = nn.LayerNorm
    
        # insert the layers
        self.linear1 = nn.Linear(in_channel, hidden, bias=bias)
        self.linear1.apply(self._init_weights)
        self.linear2 = nn.Linear(hidden, out_channel, bias=bias)
        self.linear2.apply(self._init_weights)

        self.norm1 = norm_layer(hidden)
        self.norm2 = norm_layer(out_channel)

        self.act1 = act_layer(inplace=True)
        self.act2 = act_layer(inplace=True)

        self.shortcut = None
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channel, out_channel, bias=bias),
                norm_layer(out_channel)
            )

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x
        return self.act2(out)


class TargetPred(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 m: int = 3,
                 device=torch.device("cpu")):
        super(TargetPred, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.M = m       
        self.device = device

        self.prob_mlp = nn.Sequential(
            MLP(in_channels + 2, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self.mean_mlp = nn.Sequential(
            MLP(in_channels + 2, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, feat_in: torch.Tensor, tar_candidate: torch.Tensor, candidate_mask=None):
        # dimension must be [batch size, 1, in_channels]

        batch_size, n, _ = tar_candidate.shape
        #print("test: ",feat_in.shape)
        #print("test1: ",feat_in.repeat(1, n, 1).shape)   #[batch,n,feature.shape(96)]
        # stack the target candidates to the end of input feature
        feat_in_repeat = torch.cat([feat_in.repeat(1, n, 1), tar_candidate], dim=2)#[batch,n,feature.shape(96+2)]
        # compute probability for each candidate
        prob_tensor = self.prob_mlp(feat_in_repeat).squeeze(2)
        tar_candit_prob = F.softmax(prob_tensor, dim=-1) # [batch_size, n]
        tar_offset_mean = self.mean_mlp(feat_in_repeat)    # [batch_size, n, 2]

        return tar_candit_prob, tar_offset_mean

class MotionEstimation(nn.Module):
    def __init__(self,
                 in_channels,
                 horizon=30,
                 hidden_dim=128):
        super(MotionEstimation, self).__init__()
        self.in_channels = in_channels
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        self.traj_pred = nn.Sequential(
            MLP(in_channels + 2, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, horizon * 2)
        )

    def forward(self, feat_in: torch.Tensor, loc_in: torch.Tensor):


        batch_size=loc_in.shape[0]
        M=loc_in.shape[1]
        #print("444: ",batch_size, M,feat_in.shape, loc_in.shape)
        if M > 1:
            # target candidates
            input = torch.cat([feat_in.repeat(1, M, 1), loc_in], dim=2)
        else:
            # targt ground truth
            input = torch.cat([feat_in, loc_in], dim=-1)
        return self.traj_pred(input)

class TrajScoreSelection(nn.Module):
    def __init__(self,
                 feat_channels,
                 horizon=30,
                 hidden_dim=64,
                 temper=0.01):

        super(TrajScoreSelection, self).__init__()
        self.feat_channels = feat_channels
        self.horizon = horizon
        self.temper = temper

        self.score_mlp = nn.Sequential(
            MLP(feat_channels + horizon * 2, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat_in: torch.Tensor, traj_in: torch.Tensor):

        batch_size, M, _,_ = traj_in.size()
        traj_in=traj_in.reshape(batch_size, M,-1)
        input_tenor = torch.cat([feat_in.repeat(1, M, 1), traj_in], dim=2)
        temp=self.score_mlp(input_tenor)
        out=F.softmax(temp.squeeze(-1), dim=-1)
        return  out

