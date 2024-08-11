import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import dgl
import argparse
import torch
import logging

import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader
from code_context_model.gnn import RGCN, GCN, GAT, GraphSage
from code_context_model.build_dataset import ExpandGraphDataset, split_dataset

ste2id = {'COLLABORATOR': 12, 'OTHER': 5, 'ABSTRACT': 8, 'INTERFACE': 9, 'FIELD': 5, 'COMMAND-COLLABORATOR': 20, 'MINIMAL_ENTITY': 14, 'CONSTRUCTOR': 11, 'DEGENERATE': 10, 'BOUNDARY': 8, 'FACTORY': 7, 'FACTORY-COLLABORATOR': 20, 'INCIDENTAL': 10, 'GET': 3, 'NON_VOID_COMMAND-COLLABORATOR': 29, 'DATA_PROVIDER': 13, 'ENTITY': 6, 'BOUNDARY-COMMANDER': 18, 'PROPERTY-COLLABORATOR': 21, 'GET-COLLABORATOR': 16, 'PREDICATE-COLLABORATOR': 22, 'EMPTY': 5, 'SET': 3, 'PREDICATE': 9, 'SET-COLLABORATOR': 16, 'NOTFOUND': 8, 'PROPERTY': 8, 'LOCAL_CONTROLLER': 16, 'PROPERTY-LOCAL_CONTROLLER': 25, 'NON_VOID_COMMAND': 16, 'COMMAND-LOCAL_CONTROLLER': 24, 'NON_VOID_COMMAND-LOCAL_CONTROLLER': 33, 'SET-LOCAL_CONTROLLER': 20, 'CONTROLLER': 10, 'CONSTRUCTOR-COLLABORATOR': 24, 'COMMAND': 7, 'PREDICATE-LOCAL_CONTROLLER': 26, 'DATA_CLASS': 10, 'FACTORY-LOCAL_CONTROLLER': 24, 'CONSTRUCTOR-LOCAL_CONTROLLER': 28, 'BOUNDARY-DATA_PROVIDER': 22, 'COMMANDER': 9, 'GET-LOCAL_CONTROLLER': 20, 'LAZY_CLASS': 10, 'POOL': 4, 'CONSTRUCTOR-CONTROLLER': 22, 'VOID_ACCESSOR-COLLABORATOR': 26}
id2ste = {v:k for k, v in ste2id.items()}

all_stes, all_embeddings, all_ids = [], [], []

def load_my_gnn_model_dict(model, model_path):
    old_state_dict = torch.load(model_path)
    mapping = {
        'conv1': 'conv_layers.0',
        'conv2': 'conv_layers.1',
        'conv3': 'conv_layers.2'
    }
    new_state_dict = {}
    for old_key, value in old_state_dict.items():
        for old_prefix, new_prefix in mapping.items():
            if old_key.startswith(old_prefix):
                new_key = old_key.replace(old_prefix, new_prefix)
                new_state_dict[new_key] = value

    # 加载重命名后的 state_dict 到新模型
    model.load_state_dict(new_state_dict, strict=True)

def store_logits(logits, stereotype, id):
    global ste_results

    logits_list = logits.cpu().tolist()
    ster_list = stereotype.cpu().tolist()
    id_list = id.cpu().tolist()

    all_stes.extend(ster_list)
    all_embeddings.extend(logits_list)
    all_ids.extend(id_list)


def do_clustering():
    fn = "./analysis/stereotype_logits.parquet"
    df = pd.read_parquet(fn)

    df['stereotype'] = df['stereotype_id'].apply(lambda x: id2ste[x])

    # all nodes clustering
    # cluster_eval_res = {}
    # for num_cluster in range(10, 31):
    #     embeddings = df.embeddings.to_list()
    #     kmeans = KMeans(n_clusters=num_cluster, random_state=42)
    #     kmeans.fit(embeddings)
    #     y = kmeans.predict(embeddings)
    #     silhouette_avg = silhouette_score(embeddings, y)
    #     cluster_eval_res[num_cluster] = silhouette_avg
    #     print(silhouette_avg)
    # print( [k for k, v in cluster_eval_res.items() if v==max(cluster_eval_res.values())][0] )

    interested_stes = ["COLLABORATOR","SET","PROPERTY","NON_VOID_COMMAND","GET-LOCAL_CONTROLLER", "LAZY_CLASS", "CONSTRUCTOR", "COMMAND"]
    best_cluster_num = {
        "COLLABORATOR": 0,
        "SET": 0,
        "PROPERTY": 0,
        "NON_VOID_COMMAND": 0,
        "GET-LOCAL_CONTROLLER": 0,
        "LAZY_CLASS": 0,
        "CONSTRUCTOR": 0, 
        "COMMAND": 0,

    }

    df = df[df['stereotype'].isin(interested_stes)]
    for name, group_df in df.groupby('stereotype'):
        print(f"Group: {name}")
        group_embeddings = group_df.embeddings.to_list()

        cluster_eval_res = {}
        for num_cluster in range(2, 3):
            kmeans = KMeans(n_clusters=num_cluster, random_state=42)
            kmeans.fit(group_embeddings)
            y = kmeans.predict(group_embeddings)
            # print(y)
            # 计算聚类指标
            silhouette_avg = silhouette_score(group_embeddings, y)
            cluster_eval_res[num_cluster] = silhouette_avg
        print(cluster_eval_res)
        best_cluster_num[name] =  [k for k, v in cluster_eval_res.items() if v==max(cluster_eval_res.values())][0] 
        # calinski_harabasz = calinski_harabasz_score(group_embeddings, y)
        # davies_bouldin = davies_bouldin_score(group_embeddings, y)
        # sse = kmeans.inertia_  # SSE
    
        # 打印聚类指标
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        # print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        # print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
        # print(f"SSE (Sum of Squared Errors): {sse:.4f}")
    print(best_cluster_num)
    # print(df.stereotype.value_counts())
    # print(df.info())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/data0/xiaoyez/CodeContextModel/dataset_stereotype_step1', help='input directories')
    parser.add_argument('--test_model_path', type=str, default='/data0/xiaoyez/CodeContextModel/model_output/07-14-00-19/model_48.pth', help='test model path')
    parser.add_argument('--device', type=int, default=1, help='device id')
    parser.add_argument('--do_clustering', action='store_true', help='test the model') 
    args = parser.parse_args()

    if args.do_clustering:
        do_clustering()
        exit(0)


    test_dataset = torch.load(os.path.join(args.input_dir, 'test_dataset.pt'))
    logger.info(f"Test dataset loaded, size: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=dgl.batch)

    num_dims = 1024 # bge embedding dims
    model = RGCN(in_feat=num_dims, h_feat=num_dims, gnn_layers=3, out_feat=1, num_rels=8)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # load dict
    load_my_gnn_model_dict(model, args.test_model_path)

    # do test
    logger.info("======= Start testing =======")

    model.eval()
    with torch.no_grad():
        for i, batch_graphs in enumerate(test_loader):
            batch_graphs = batch_graphs.to(device)
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['label'] = batch_graphs.edata['label'].to(device)
            batch_graphs.ndata['label'] = batch_graphs.ndata['label'].to(device)
            logits = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['label'].squeeze(1))

            store_logits(logits, batch_graphs.ndata['stereotype'], batch_graphs.ndata['id'])
        
        logger.info(f"Test finished")
    
    # store global ste_results to to_parquet
    df = pd.DataFrame({
        'file_id': [ids[0] for ids in all_ids],
        'vertex_id': [ids[1] for ids in all_ids],
        'stereotype_id': all_stes,
        'embeddings': all_embeddings,
    })
    df.to_parquet("./analysis/stereotype_logits.parquet")

    
    
    # for ster, logit in zip(ster_list, logits_list):
    #     ste_results[id2ste[ster]].append(logit)