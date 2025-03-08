import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from layer import GraphConv‘
from enum import Enum
from dgl.nn import GraphConv

# node label 和 code_context_model/build_dataset.py 中的保持一致
class NodeLabel(Enum):
    SEED = -1
    CONTEXT = 1
    NON_CONTEXT = 0
    NEG_CONTEXT = 2

class GCNRec(nn.Module):

    def __init__(self, vocab_sz, pretrain_emb,
                 dropout=0.2, margin=1, emb_dim=64,
                 kernel_dim=128):
        super(GCNRec, self).__init__()
        self.margin = margin
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.kernel_dim = kernel_dim
        # self.lookup_index = lookup_index

        self.word_emb = nn.Embedding(vocab_sz+1, emb_dim, padding_idx=0)
        self.word_emb.from_pretrained(pretrain_emb)

        self.a = nn.Parameter(torch.Tensor(2*emb_dim))
        self.word_trans = nn.Linear(emb_dim, 2*emb_dim)
        self.rnn = nn.GRU(emb_dim, emb_dim, num_layers=2,
                          dropout=0.2, batch_first=True)
        
        self.other_pos_emb = nn.Embedding(vocab_sz+1, emb_dim)
        self.user_pos_emb = nn.Embedding(vocab_sz+1, emb_dim)
        self.item_pos_emb = nn.Embedding(vocab_sz+1, emb_dim)
        self.conv1 = GraphConv(emb_dim, kernel_dim)
        self.conv2 = GraphConv(kernel_dim, kernel_dim)
        self.linear = nn.Linear(2*kernel_dim, 2*emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(2*self.emb_dim)
        self.a.data.uniform_(-stdv, stdv)

    def att_pooling(self):
        emb = self.word_emb(self.lookup_index)
        e = self.word_trans(emb)
        # (node_sz, seq_len, 128) -> (node_sz, seq_len, 1)
        attn = torch.softmax(e.mul(self.a).sum(2), dim=1).unsqueeze(2)
        # (node_sz, seq_len, 64) -> (node_sz, 64)
        out = emb.mul(attn).sum(1)
        return out
    
    def rnn_encoding(self, embedding_index):
        # (node_sz, seq_len, 64)
        print(f"embedding_index shape: {embedding_index.shape}")
        print(f"embedding_index: {embedding_index}")
        print(f"max: {torch.max(embedding_index).item()}")
        print(f"vocab_sz: {self.word_emb.weight.shape[0]}")
        print(f"{self.word_emb.weight.shape}")

        emb = self.word_emb(embedding_index)
        # print(f"emb shape: {emb.shape}")
        # # Create mask for non-zero elements (assuming padding_idx=0)
        # non_zero_mask = (embedding_index != 0).unsqueeze(-1)  # (node_sz, seq_len, 1)
        # # Calculate mean only for non-zero elements
        # sum_embeddings = (emb * non_zero_mask).sum(dim=1)  # (node_sz, emb_dim)
        # count_non_zero = non_zero_mask.sum(dim=1).clamp(min=1)  # (node_sz, 1)
        # print(f"count_non_zeros shape: {count_non_zero.shape}")
        # emb = sum_embeddings / count_non_zero  # (node_sz, emb_dim)
        # print(f"emb shape: {emb.shape}")
        # print(f"emb: {emb[0]}")
        rnn_out, hidden = self.rnn(emb)
        # (node_sz, 64)
        return hidden[-1]
    
    def _get_mean_embeddings(self, node_type, emb_layer, embedding_index, node_labels):
            """Helper function to calculate mean embeddings for given node type"""
            # 添加索引检查
            mask = (node_labels == node_type)
            valid_indices = embedding_index[mask]
            if torch.any(valid_indices >= len(emb_layer)):
                print("Warning: Invalid indices detected")
                valid_indices = torch.clamp(valid_indices, 0, len(emb_layer)-1)
            
            embeddings = emb_layer[valid_indices]
            non_zero_mask = (valid_indices != 0).unsqueeze(-1)
            sum_embeddings = (embeddings * non_zero_mask).sum(dim=1)  # (num_nodes, emb_dim)
            count_non_zero = non_zero_mask.sum(dim=1).clamp(min=1)  # (num_nodes, 1)
            
            return sum_embeddings / count_non_zero, mask
    
    def refine_embedding(self, graph, node_labels, embedding_index):
        # embedding_index: (node_sz, seq_len)
        # (node_sz, seq_len, emb_dim) -> (node_sz, emb_dim)
        pos_emb = torch.zeros((len(node_labels), self.emb_dim), 
                            device=embedding_index.device)
        
        def set_pos_emb_by_type(node_type, pos_emb_by_type):
            mean_emb, mask = self._get_mean_embeddings(node_type, pos_emb_by_type, embedding_index, node_labels)
            pos_emb[mask] = mean_emb

        set_pos_emb_by_type(NodeLabel.SEED.value, self.user_pos_emb.weight)
        set_pos_emb_by_type(NodeLabel.CONTEXT.value, self.item_pos_emb.weight)
        set_pos_emb_by_type(NodeLabel.NON_CONTEXT.value, self.other_pos_emb.weight)
        set_pos_emb_by_type(NodeLabel.NEG_CONTEXT.value, self.other_pos_emb.weight)

        print(f"pos_emb shape: {pos_emb.shape}")
        all_emb = pos_emb + self.rnn_encoding(embedding_index)
        h_emb = []
        conv_emb = F.dropout(self.conv1(graph, all_emb),
                             p=self.dropout, training=self.training)
        h_emb.append(conv_emb)
        conv_emb = F.dropout(self.conv2(graph, conv_emb),
                             p=self.dropout, training=self.training)
        h_emb.append(conv_emb)
        out_emb = torch.cat(h_emb, dim=1)
        out_emb = self.linear(out_emb)
        # 根据node_type分别返回对应的embedding
        return out_emb

    def get_top_items(self, graph, embedding_index, node_labels, k):
        out_emb = self.refine_embedding(graph, node_labels, embedding_index)

        user_idx = (node_labels == -1).nonzero().squeeze()
        user_x = F.embedding(user_idx, out_emb)
        user_x = user_x.unsqueeze(0) if len(user_x.shape) == 1 else user_x # (batch_sz, 1, emb_dim)
        # print(f"user_x shape: {user_x.shape}")

        # 计算 nodel_lables == 0 或 1 的 节点的平均embedding
        candidates_idx = ((node_labels == 0) | (node_labels == 1) | (node_labels == 2)).nonzero().squeeze()
        # print(f"node_labels shape: {node_labels.shape}")
        # print(f"candidates_idx shape: {candidates_idx.shape}")
        candidates_emb = F.embedding(candidates_idx, out_emb) # # (batch_sz, #candidates, emb_dim)
        # print(f"candidates_emb shape: {candidates_emb.shape}")

        ratings = candidates_emb.mm(user_x.transpose(-2, -1)).squeeze(-1) # (batch_sz, #candidates)
        # print(f"ratings shape: {ratings.shape}")

        k = min(k, len(ratings))
        # print(f"k: {k}")
        values, indices = ratings.topk(k) 
        # print(f"indices shape: {indices.shape}")
        return indices, ratings

    def get_non_zero_embedding(items, item_x):
        non_zero_mask = (items != 0).unsqueeze(-1)  # (2, seq_len, 1)
        sum_embeddings = (item_x  * non_zero_mask).sum(dim=1)  # (2, emb_dim)
        count_non_zero = non_zero_mask.sum(dim=1).clamp(min=1)  # (2, 1)
        item_x = sum_embeddings / count_non_zero  # (2, emb_dim)
        return item_x
    
    def forward(self, graph, embedding_index, node_labels, edge_lable):
        """
        :param user: (batch_sz,) int
        :param pos_item: (batch_sz,) int
        :param neg_item: (k*batch_sz,) int
        :param adj: laplacian matrix
        :return: loss
        """
        out_emb = self.refine_embedding(graph, node_labels, embedding_index)
        print(f"out_emb shape: {out_emb.shape}")
        # 选出eed作为user
        user_idx = (node_labels == NodeLabel.SEED.value).nonzero().squeeze()
        user_x = F.embedding(user_idx, out_emb)
        user_x = user_x.unsqueeze(0) if len(user_x.shape) == 1 else user_x
        # print(f"user_x shape: {user_x.shape}")

        pos_idx = (node_labels == NodeLabel.CONTEXT.value).nonzero().squeeze()
        pos_item_x = F.embedding(pos_idx, out_emb)  # (2, seq_len, emb_dim) 最终获得的嵌入表示
        pos_item_x = pos_item_x.unsqueeze(0) if len(pos_item_x.shape) == 1 else pos_item_x
        # print(f"pos_item_x shape: {pos_item_x.shape}")

        # node_labels == 2的节点是负样本
        neg_idxes = (node_labels == NodeLabel.NEG_CONTEXT.value).nonzero().squeeze()
        neg_item_x = F.embedding(neg_idxes, out_emb)  # (2, seq_len, emb_dim) 最终获得的嵌入表示
        # print(f"neg_item_x shape: {neg_item_x.shape}")
       
        # inner product between user and pos_item (batch_sz,)
        pos_score = torch.matmul(user_x, pos_item_x.transpose(0, 1))
        pos_score = torch.diagonal(pos_score)
        # print(f"pos_score shape: {pos_score.shape}")

        # (k, batch_sz, emb_dim)
        neg_item_x = neg_item_x.view(-1, user_x.size()[0], user_x.size(1))
        # (k, batch_sz, emb_dim)x(batch_sz, emb_dim)->(k, batch_sz, emb_dim)
        # sum(2) -> (k, batch_sz)
        neg_score = neg_item_x.mul(user_x).sum(2)
        # (k, batch_sz) - (batch_sz,) -> (k, batch_sz)
        # expectation of negative samples: mean(0) -> (batch_sz,)
        # total loss: sum() -> (scalar)
        diff = neg_score - pos_score + self.margin
        rank_loss = torch.mean(diff, dim=0).clamp(min=1e-6, max=1e4).sum()
        #ce_loss = self.cross_entropy_loss(pos_score, neg_score, label)
        loss = rank_loss #self.log_loss(pos_score, neg_score)
        return loss

    def log_loss(self, pos_score, neg_score):
        logits = torch.mean(pos_score - neg_score, dim=0)
        return -torch.sum(torch.log(torch.sigmoid(logits)))

    def cross_entropy_loss(self, pos_score, neg_score, label):
        logits = torch.cat([pos_score, torch.flatten(neg_score)])
        loss = F.binary_cross_entropy_with_logits(logits, label.float())
        return loss

