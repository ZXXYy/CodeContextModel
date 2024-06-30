import os
import dgl
import torch
import os
import random
import logging
import pandas as pd
import numpy as np

from dgl.data import DGLDataset
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger('dataset')

# one-hot encoding for edge labels
# edge_label = {
#     "declares": [0,0,0,1],
#     "calls": [0,0,1,0],
#     "inherits": [0,1,0,0],
#     "implements": [1,0,0,0],
# }
edge_label = {
    "declares": 0,
    "calls": 1,
    "inherits": 2,
    "implements": 3,
}

class ExpandGraphDataset(DGLDataset):
    def __init__(self, xml_files, embedding_dir, device, debug=False):
        
        self.xml_files = xml_files[:512] if debug else xml_files
        self.embedding_dir = embedding_dir
        self.device = device

        super().__init__(name='expand_graph_dataset')

    def process(self, embedding_model='BgeEmbedding'):
        logger.info(f"building dataset from xml filesm len: {len(self.xml_files)}...")
        self.graphs = []
        for xml_file in tqdm(self.xml_files):
            model_dir = xml_file.split('/')[-3]
            embedding_path = os.path.join(self.embedding_dir, f"{model_dir}_{embedding_model}_embedding.pkl")
            # load embedding
            with open(embedding_path, 'rb') as f:
                df_embeddings = pd.read_pickle(f)
            # 解析XML文件
            tree = ET.parse(xml_file)
            root = tree.getroot()
            graph_element = root.find('graph')

            # 获取顶点
            vertices = graph_element.find('vertices')
            vertex_ids = [int(vertex.get('id')) for vertex in vertices.findall('vertex')]
            id_map = {v: i for i, v in enumerate(vertex_ids)}
            vertex_features = []
            vertex_labels = []
            for vertex in vertices.findall('vertex'):
                node_id = '_'.join([model_dir, vertex.get('kind'), vertex.get('ref_id')]) 
                node_embedding = list(df_embeddings[df_embeddings['id'] == node_id]['embedding'])
                vertex_features.append(node_embedding)
                if vertex.get('origin', '0') == '1' and vertex.get('seed', '0') == '0':
                    vertex_labels.append(1)
                else:
                    vertex_labels.append(0)

            # print(vertex_features)
            logger.debug(f"Read vertex_features features len: {len(vertex_features)}")
            # 将顶点特征转换为张量
            node_features = torch.tensor(vertex_features)
            node_labels = torch.tensor(vertex_labels)

            # 获取边
            edges = graph_element.find('edges')
            edge_list = []
            edge_labels = []
            for edge in edges.findall('edge'):
                start = int(edge.get('start'))
                end = int(edge.get('end'))
                label = edge.get('label')
                # 添加无向边
                edge_list.append((id_map[start], id_map[end]))
                edge_list.append((id_map[end], id_map[start]))
                edge_labels.append(edge_label[label] )
                edge_labels.append(edge_label[label] + 4) # 反向边

            # 将边特征转换为张量
            edge_features = torch.tensor(edge_labels).unsqueeze(1) # 转换为2D张量

            # 构建DGL图
            src, dst = zip(*edge_list)
            g = dgl.graph(data=(src, dst)).to(self.device)

            # 设置节点特征
            g.ndata['feat'] = node_features.to(self.device)
            # 设置节点标签
            g.ndata['label'] = node_labels.to(self.device)
            # 设置边特征
            g.edata['label'] = edge_features.to(self.device)

            self.graphs.append(g)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
    
def split_dataset(dataset, train_ratio=0.8, valid_ratio=0.1):
    # 生成数据集索引
    indices = np.arange(len(dataset))
    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(indices, test_size=0.2, train_size=0.8, random_state=42)

    # 划分训练集和验证集
    train_indices, valid_indices = train_test_split(train_indices, test_size=0.1, train_size=0.9, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    xml_dir = '/data0/xiaoyez/CodeContextModel/data/repo_first_3/60/seed_expanded'
    xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith('.xml')]
    # xml_files = ['data/mylyn/60/seed_expanded/1_step_seeds_3_expanded_model.xml']
    print(xml_files)
    data_builder = ExpandGraphDataset(xml_files, "/data0/xiaoyez/CodeContextModel/bge_embedding_results", torch.device('cpu'), debug=True)
    # 切分数据集
    train_dataset, valid_dataset, test_dataset = split_dataset(data_builder)
    # 使用 DataLoader 加载子集
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)

    logger.info(f"train dataset: {len(train_dataset)}")
    for batch_graphs in train_loader:
        print(batch_graphs)
    logger.info(f"train dataset: {len(valid_dataset)}")
    for batch_graphs in valid_loader:
        print(batch_graphs)
    logger.info(f"train dataset: {len(test_dataset)}")
    for batch_graphs in test_loader:
        print(batch_graphs)