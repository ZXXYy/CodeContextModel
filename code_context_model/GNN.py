import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import RelGraphConv

class RGCN(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, num_rels):
        super(RGCN, self).__init__()
        self.conv1 = RelGraphConv(in_feat, h_feat, num_rels, regularizer='basis', num_bases=4)
        self.conv2 = RelGraphConv(h_feat, h_feat, num_rels, regularizer='basis', num_bases=4)
        self.conv3 = RelGraphConv(h_feat, h_feat, num_rels, regularizer='basis', num_bases=4)
        self.fc = nn.Linear(h_feat, out_feat)

    def _initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, RelGraphConv):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, g, feat, etype):
        h = self.conv1(g, feat, etype)
        h = F.relu(h)
        h = self.conv2(g, h, etype)
        h = F.relu(h)
        h = self.conv3(g, h, etype)
        h = F.relu(h)
        h = self.fc(h)
        return F.softmax(h, dim=1)
    

