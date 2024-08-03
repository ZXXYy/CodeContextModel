import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import RelGraphConv, GATConv, SAGEConv
from dgl.nn import GraphConv


class MLP(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_feat, h_feat//2)
        self.fc2 = nn.Linear(h_feat//2, h_feat//4)
        self.fc3 = nn.Linear(h_feat//4,out_feat)

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))



class RGCN(nn.Module):
    def __init__(self, in_feat, h_feat, gnn_layers=3, out_feat=1, num_rels=8):
        """    
        Args:
            # in_feats: 输入特征的维度(embedding dims)
            # h_feats: 隐层特征的维度
            # out_feats: 输出特征的维度
            # num_rels: 关系的数量
        """
        super(RGCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(RelGraphConv(in_feat, h_feat, num_rels, regularizer='basis', num_bases=4))
        for i in range(gnn_layers - 1):
            self.conv_layers.append(RelGraphConv(h_feat, h_feat, num_rels, regularizer='basis', num_bases=4))
        # self.mlp = MLP(h_feat, h_feat, out_feat)      
        
        self._initialize_parameters() 

    def _initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, RelGraphConv)):
                for param in m.parameters():
                    if param.dim() > 1:
                        nn.init.kaiming_uniform_(param, nonlinearity='relu')
                    else:
                        nn.init.zeros_(param)


    def forward(self, g, feat, etype):
        h_prev = feat
        for i, layer in enumerate(self.conv_layers):
            h = layer(g, h_prev, etype)
            h = F.relu(h) + h_prev # 添加残差连接
            h_prev = h  # 保存当前层的输出以用作下一层的残差连接
            
        return h_prev


        # h = self.conv1(g, feat, etype)
        # h = F.relu(h) + feat  # 添加残差连接
        # h_prev = h  # 保存当前层的输出以用作下一层的残差连接

        # h = self.conv2(g, h, etype)
        # h = F.relu(h) + h_prev  # 添加残差连接
        # h_prev = h  # 更新h_prev

        # h = self.conv3(g, h, etype)
        # h = F.relu(h) + h_prev  # 添加残差连接
        # # h = F.relu(h)
        # # return self.mlp(h)
        # return h
        # # return F.sigmoid(h)
    

class GCN(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat=1):
        """    
        Args:
            # in_feats: 输入特征的维度(embedding dims)
            # h_feats: 隐层特征的维度
            # out_feats: 输出特征的维度
        """
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feat, h_feat,  norm='both')
        self.conv2 = GraphConv(h_feat, h_feat, norm='both')
        self.conv3 = GraphConv(h_feat, h_feat, norm='both')

    def forward(self, g, feat, etype):
        h = self.conv1(g, feat)
        h = F.relu(h) + feat  # 添加残差连接
        h_prev = h  # 保存当前层的输出以用作下一层的残差连接

        h = self.conv2(g, h)
        h = F.relu(h) + h_prev  # 添加残差连接
        h_prev = h  # 更新h_prev

        h = self.conv3(g, h)
        h = F.relu(h) + h_prev  # 添加残差连接

        return h


class GAT(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat=1, num_heads=4):
        """    
        Args:
            # in_feats: 输入特征的维度(embedding dims)
            # h_feats: 隐层特征的维度
            # out_feats: 输出特征的维度
        """
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feat, h_feat, num_heads=num_heads, attn_drop=0.3)
        self.conv2 = GATConv(h_feat * num_heads, h_feat, num_heads=num_heads, attn_drop=0.3)
        self.conv3 = GATConv(h_feat * num_heads, h_feat, num_heads=1, attn_drop=0.3)

    def forward(self, g, feat, etype):
        h = self.conv1(g, feat)
        h = h.flatten(1)  # 将多头输出展平
        h = F.relu(h) 

        h = self.conv2(g, h)
        h = h.flatten(1)  # 将多头输出展平
        h = F.relu(h) 

        h = self.conv3(g, h)
        h = h.mean(1)  # 将多头输出平均
        # h = F.relu(h)
        return h

class GraphSage(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat=1):
        """    
        Args:
            # in_feats: 输入特征的维度(embedding dims)
            # h_feats: 隐层特征的维度
            # out_feats: 输出特征的维度
        """
        super(GraphSage, self).__init__()
        self.conv1 = SAGEConv(in_feat, h_feat, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feat, h_feat, aggregator_type='mean')
        self.conv3 = SAGEConv(h_feat, h_feat, aggregator_type='mean')
    
    def forward(self, g, feat, etype):
        h = self.conv1(g, feat)
        h = F.relu(h) + feat  # 添加残差连接
        h_prev = h  # 保存当前层的输出以用作下一层的残差连接

        h = self.conv2(g, h)
        h = F.relu(h) + h_prev  # 添加残差连接
        h_prev = h  # 更新h_prev

        h = self.conv3(g, h)
        h = F.relu(h) + h_prev  # 添加残差连接
        # h = F.relu(h)
        return h