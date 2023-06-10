import copy
import torch
from torch import nn

import config
from network.mlp import MLP
from network.global_graph import GlobalGraph
from network.sub_graph import SubGraph
from network.util import TargetPred
from network.util import MotionEstimation,TrajScoreSelection
import config
import torch.nn.functional as F
import matplotlib.pyplot as plt


class VectorNetWithPredicting(nn.Module):

    def __init__(self, v_len, time_stamp_number, data_dim=2):

        super(VectorNetWithPredicting, self).__init__()
        self.vector_net = VectorNet(v_len).to(config.device)
        self.traj_decoder = MLP(input_size=self.vector_net.p_len*2,
                                output_size=time_stamp_number * data_dim).to(config.device)
        self.dim = data_dim
        self.target_pred_layer = TargetPred(
            in_channels=192,
            hidden_dim=64,
            m=5,
        )
        self.motion_estimator = MotionEstimation(
            in_channels=self.vector_net.p_len*2,
            horizon=30,
            hidden_dim=64
        )

        self.traj_score_layer = TrajScoreSelection(
            feat_channels=192,
            horizon=30,
            hidden_dim=64,
            temper=10.0
        )
        self.m = 5

    def forward(self, item_num, target_id, polyline_list, cand_goals, goals, epoch):

        # [batch_size,feature.shape(96)]
        
        feature = self.vector_net(item_num, target_id, polyline_list, epoch)

        if torch.isnan(feature[0,0]):
            print("111111111111111")
        batch_size = feature.shape[0]
        N = cand_goals.shape[1]
        feature = feature.unsqueeze(1)  # [batch_size,1,feature.shape(96)]
        target_prob, offset = self.target_pred_layer(feature,cand_goals.to(torch.float32),[])
        target_gt = goals.view(-1,1,2)

        traj_with_gt = self.motion_estimator(
            feature, target_gt.float())
       # traj_with_gt = self.traj_decoder(feature)
        _, indices = target_prob.topk(self.m, dim=1)

        batch_idx = torch.vstack([torch.arange(0, batch_size).to(config.device) for _ in range(self.m)]).T
        target_pred_se, offset_pred_se = cand_goals[batch_idx,
                                                    indices], offset[batch_idx, indices]
        # print("dd1:", target_pred_se.shape,
        #       offset_pred_se.shape, feature.shape)
        trajs = self.motion_estimator(feature, target_pred_se.to(
            torch.float32) + offset_pred_se.to(torch.float32))
        trajs = trajs.reshape(batch_size, self.m, -1, self.dim)
        traj_with_gt = traj_with_gt.reshape(batch_size, -1, self.dim)
        score = self.traj_score_layer(feature, trajs)

        mask =torch.zeros(batch_size,cand_goals.shape[1]).cuda()
        mask[:,0]=1.0

        #print("mask: ",mask_target_p_select.shape,mask.shape)
        return {
            "traj_with_gt": traj_with_gt,
            "trajs": trajs,
            "score": score,
            "target_select": target_prob,
            "offset":offset,
            "mask_target_p_select":mask,
        }


class VectorNet(nn.Module):

    def __init__(self, v_len, sub_layers=3, global_layers=1):

        super(VectorNet, self).__init__()
        self.sub_graph_map = SubGraph(
            layers_number=sub_layers, v_len=v_len).to(config.device)
        self.sub_graph_other_traj = SubGraph(
            layers_number=sub_layers, v_len=v_len).to(config.device)
        self.sub_graph_ego_traj = SubGraph(
            layers_number=sub_layers, v_len=v_len).to(config.device)
        self.p_len = v_len * (2 ** sub_layers)

        self.global_graph = GlobalGraph(
            len=self.p_len, layers_number=global_layers).to(config.device)

    def forward(self, item_num, target_id, polyline_list, epoch):

        assert target_id.dtype == torch.int64

        batch_size = item_num.shape[0]
        p_list = []
        cnt = 0
        for polyline in polyline_list:
            if cnt < 1:
                polyline0 = polyline.to(config.device)
                temp_debug_polyline = polyline0.cpu().numpy()
                polyline0 = torch.cat([polyline0], axis=0)
                p = self.sub_graph_ego_traj(polyline0).to(
                    config.device)  # [batch_size, p_len]
            elif cnt < 30:
                polyline1 = polyline.to(config.device)
                polyline1 = torch.cat([polyline1], axis=0)
                p = self.sub_graph_other_traj(polyline1).to(
                    config.device)  # [batch_size, p_len]
            else:
                polyline2 = polyline.to(config.device)
                polyline2 = torch.cat([polyline2], axis=0)
                p = self.sub_graph_map(polyline2).to(
                    config.device)  # [batch_size, p_len]
            p_list.append(p)
            cnt = cnt+1

        P = torch.stack(p_list, axis=1)  # [batch_size, p_number, p_len]
        assert P.shape == (batch_size, len(polyline_list), self.p_len)
        feature = self.global_graph(P, target_id)  # [batch_size, p_len]
        feature = torch.cat([feature, P[:, 0, :]], dim=-1)
        return feature
