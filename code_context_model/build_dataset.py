import os
import dgl
import torch
import os
import shutil
import random
import logging
import argparse

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from dgl.data import DGLDataset
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
    def __init__(self, xml_files, embedding_dir, embedding_model, debug=False):
        
        self.xml_files = xml_files[:64] if debug else xml_files
        self.embedding_dir = embedding_dir
        self.embedding_model = embedding_model

        super().__init__(name='expand_graph_dataset')

    def process(self):
        logger.info(f"building dataset from xml files len: {len(self.xml_files)}...")
        self.graphs = []
        for xml_file in tqdm(self.xml_files):
            model_dir = xml_file.split('/')[-3]
            embedding_path = os.path.join(self.embedding_dir, f"{model_dir}_{self.embedding_model}_embedding.pkl")
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
                if self.embedding_model == 'codebert':
                    node_embedding = list(df_embeddings[df_embeddings['id'] == node_id]['embedding'])
                    node_embedding = node_embedding[0].tolist()
                else:
                    node_embedding = list(df_embeddings[df_embeddings['id'] == node_id]['embedding'])
                    node_embedding = node_embedding[0]
                vertex_features.append(node_embedding)
                if vertex.get('seed', '0') == '1':
                    vertex_labels.append(-1)
                else:
                    if vertex.get('origin', '0') == '1':
                        vertex_labels.append(1)
                    else:
                        vertex_labels.append(0)

            # print(vertex_features)
            # logger.info(f"Read vertex_features features len: {len(vertex_features)}")
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
            g = dgl.graph(data=(src, dst))

            # 设置节点特征
            try:
                g.ndata['feat'] = node_features
                # 设置节点标签
                g.ndata['label'] = node_labels
                # 设置边特征
                g.edata['label'] = edge_features
                self.graphs.append(g)
            except Exception as e:
                logger.info(f"xml_files: {xml_file}")
                logger.info(f"node_features: {node_features.shape}")
                logger.info(f"nodes: {len(vertices)}")

            

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.graphs[i] for i in idx]
        elif isinstance(idx, np.ndarray):
            return [self.graphs[i] for i in idx.tolist()]
        else:
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

def read_xml_dataset(data_dir):
    result_xmls = []
    dir_names = os.listdir(data_dir)
    
    for dir_name in dir_names:
        # dataset_dir = os.path.join(data_dir, dir_name, "1_step_collaspe")
        # if not os.path.exists(dataset_dir):
        #     continue
        # for file_name in os.listdir(dataset_dir):
        #     if file_name.endswith(".xml"):
        #         result_xmls.append(os.path.join(dataset_dir, file_name))
        # file_names = os.listdir(dataset_dir)
        # step_1_files = [f for f in file_names if f.startswith("collapse_1_step")]
        step_1_dir = os.path.join(data_dir, dir_name, "1_step_seeds_cc")
        if not os.path.exists(step_1_dir):
            continue
        for step_1_file in os.listdir(step_1_dir):
            logger.debug(f"Step 1 file: {step_1_file}")
            if os.path.exists(os.path.join(step_1_dir, step_1_file)):
                result_xmls.append(os.path.join(step_1_dir, step_1_file))
    
    logger.info(f"Read total xml files: {len(result_xmls)}")
    return result_xmls

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='data', help='input directory')
    parser.add_argument('--embedding_dir', type=str, default='data', help='embedding directory')
    parser.add_argument('--output_dir', type=str, default='dataset', help='dataset output directory')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    
    args = parser.parse_args()

    xml_files = read_xml_dataset(args.input_dir)
    data_builder = ExpandGraphDataset(xml_files=xml_files, embedding_dir=args.embedding_dir, embedding_model='BgeEmbedding', debug=args.debug)
    train_dataset, valid_dataset, test_dataset = split_dataset(data_builder)

    logger.info(f"train dataset: {len(train_dataset)}")
    logger.info(f"valid dataset: {len(valid_dataset)}")
    logger.info(f"test dataset: {len(test_dataset)}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # write the dataset to disk
    torch.save(train_dataset, os.path.join(args.output_dir, 'train_dataset.pt'))
    torch.save(valid_dataset, os.path.join(args.output_dir, 'valid_dataset.pt'))
    torch.save(test_dataset,  os.path.join(args.output_dir, 'test_dataset.pt'))

    # 使用 DataLoader 加载子集
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)    
    # file_path = '/data2/shunliu/pythonfile/code_context_model_prediction/params_validation/git_repo_code/my_mylyn/repo_first_3/1005/1_codebert_embedding.pkl'
    # df_embeddings = pd.read_pickle(file_path)
    # print(df_embeddings)


    # source_dir = "/data2/shunliu/pythonfile/code_context_model_prediction/params_validation/git_repo_code/my_mylyn/repo_first_3"
    # target_dir = "codebert_embedding_results"
    # # 确保目标目录存在
    # os.makedirs(target_dir, exist_ok=True)
    # # 递归查找并复制文件
    # for id_folder in tqdm(os.listdir(source_dir)):
    #     source_file = os.path.join(source_dir, id_folder, "1_codebert_embedding.pkl")
    #     if not os.path.exists(source_file):
    #         continue
    #     # rename the file
    #     target_file = os.path.join(target_dir, f"{id_folder}_codebert_embedding.pkl")
    #     shutil.copy(source_file, target_file)
    #     print(f"Copied: {source_file} to {target_dir}")