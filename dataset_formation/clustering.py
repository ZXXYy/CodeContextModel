import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import torch
import dgl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from code_context_model.build_dataset import ExpandGraphDataset, split_dataset
from code_context_model.train import read_xml_dataset

# clustering nodes in the graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dir', type=str, default='data', help='embedding directory')
    parser.add_argument('--device', type=int, default=1, help='device id')
    
    args = parser.parse_args()
    global device
    device = "cpu"

    xml_files = read_xml_dataset('/data0/xiaoyez/CodeContextModel/data/repo_first_3')
    data_builder = ExpandGraphDataset(xml_files=xml_files, embedding_dir=args.embedding_dir, embedding_model='BgeEmbedding', device=device, debug=True)
    train_loader = DataLoader(data_builder, batch_size=1, shuffle=True, collate_fn=dgl.batch)

    # 根据graph.ndata['feat']聚类
    for i, graph in enumerate(train_loader):
        print(graph.ndata['feat'].shape)
        # 1. 使用KMeans对节点进行聚类
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(graph.ndata['feat'])
        y_kmeans = kmeans.predict(graph.ndata['feat'])

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(graph.ndata['feat'])
        # 2. 可视化聚类结果
        plt.figure()  # 每次循环开始时创建一个新的图形
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, s=50, cmap='viridis')
        # get labels
        label_indices = (graph.ndata['label'] == -1).nonzero().squeeze().tolist()
        if type(label_indices) is not list:
            label_indices = [label_indices]
        for label_index in label_indices:
            plt.scatter(graph.ndata['feat'][label_index, 0], graph.ndata['feat'][label_index, 1], c='red', s=100, alpha=0.75, marker='X')
        temp = (graph.ndata['label'] == 1).nonzero().squeeze().tolist()
        print(temp)
        plt.scatter(X_pca[temp, 0], X_pca[temp, 1], c='green', s=100, alpha=0.75, marker='X')
        plt.savefig(f'cluster_{i}.png')
        if i>10:
            break
        # 3. 将聚类结果写入文件

    
    
