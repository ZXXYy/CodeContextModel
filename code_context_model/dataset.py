import dgl
import torch
import os
import random
import pandas as pd
import logging

from dgl.data import DGLDataset
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger('dataset')

edge_label = {
    "declares": 0,
    "calls": 1,
    "inherits": 2,
    "implements": 3,
}

class ExpandGraphDataset(DGLDataset):
    def __init__(self, xml_files):
        self.xml_files = xml_files
        super().__init__(name='expand_graph_dataset')

    def process(self, embedding_model='BgeEmbedding'):
        logger.info("building dataset...")
        self.graphs = []
        for xml_file in self.xml_files:
            model_dir = xml_file.split('/')[-3]
            embedding_path = f"data/bge_embedding_results/{model_dir}_{embedding_model}_embedding.pkl"
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
                vertex_labels.append(bool(vertex.get('origin')) and not bool(vertex.get('seed')))

            # print(vertex_features)
            logger.info(f"features len: {len(vertex_features)}")
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
                edge_list.append((id_map[start], id_map[end]))
                edge_labels.append(edge_label[label])

            # 将边特征转换为张量
            edge_features = torch.tensor(edge_labels).unsqueeze(1)  # 转换为2D张量

            # 构建DGL图
            src, dst = zip(*edge_list)
            g = dgl.graph((src, dst))

            # 设置节点特征
            g.ndata['feat'] = node_features
            # 设置节点标签
            g.ndata['label'] = node_labels
            # 设置边特征
            g.edata['label'] = edge_features

            self.graphs.append(g)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
    

if __name__ == '__main__':
    xml_dir = 'data/mylyn/60/seed_expanded'
    # xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith('.xml')]
    xml_files = ['data/mylyn/60/seed_expanded/1_step_seeds_3_expanded_model.xml']
    print(xml_files)
    data_builder = ExpandGraphDataset(xml_files)