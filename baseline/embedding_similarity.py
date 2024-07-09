import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import dgl
import logging

from torch.utils.data import DataLoader
from tqdm import tqdm

from code_context_model.train import read_xml_dataset, cosine_similarity, set_seed
from code_context_model.build_dataset import ExpandGraphDataset, split_dataset

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)


def compute_metrics(batch_logits, batch_labels, batch_num_nodes):
    batch_size = len(batch_num_nodes)
    start_idx = 0
    total_hit = {
        'top1_hit': 0,
        'top2_hit': 0,
        'top3_hit': 0,
        'top4_hit': 0,
        'top5_hit': 0,
        'top1_precision': 0,
        'top2_precision': 0,
        'top3_precision': 0,
        'top4_precision': 0,
        'top5_precision': 0,
        'top1_recall': 0,
        'top2_recall': 0,
        'top3_recall': 0,
        'top4_recall': 0,
        'top5_recall': 0,
    }
    logger.debug(f"Batch num:{batch_num_nodes}")
    for k in range(batch_size):
        labels = batch_labels[start_idx : start_idx + batch_num_nodes[k]]
        logits = batch_logits[start_idx : start_idx + batch_num_nodes[k]]
        seed_indices = (labels == -1).nonzero()
        positive_indices = (labels == 1).nonzero()
        seed_embeddings = logits[seed_indices]
        non_seed_indices = (labels != -1).nonzero()
        non_seed_embeddings = logits[non_seed_indices]

        similarities = []
        logger.debug(f"num seed indices: {len(seed_indices)}, num non seed indices: {len(non_seed_indices)}")
        logger.debug(f"num seed embeddings: {len(seed_embeddings)}, num non seed embeddings: {len(non_seed_embeddings)}")
        for i in range(0, len(seed_embeddings)):
            temp = torch.tensor([]).to(device)
            for embedding in non_seed_embeddings:
                temp = torch.cat((temp, cosine_similarity(seed_embeddings[i], embedding)))
            logger.debug(f"similarities for seed_embeddings {i}: {temp}")
            similarities = similarities + [temp]
        if len(similarities) == 0:
            start_idx += batch_num_nodes[k]
            continue
        logger.debug(f"Similarities: {similarities}")
        similarities = torch.sum(torch.stack(similarities), dim=0)
        logger.debug(f"Similarities: {similarities}")
        logger.debug(f"similarities type: {similarities.shape}")
        # find top3 similar embeddings
        for topk in range(1,6):
            temp = topk
            topk = topk if len(non_seed_indices) >= topk else len(non_seed_indices)
            topk_indices = torch.topk(similarities, topk).indices
            logger.debug(f"Top{topk} indices: {topk_indices}")
            hit = 0
            cnt = 0
            for i in range(0, len(topk_indices)):
                idx = non_seed_indices[topk_indices[i]] # get the index of the top3 embeddings
                if labels[idx] == 1:
                    hit = 1
                    cnt += 1
            total_hit[f"top{temp}_hit"] += hit
            total_hit[f"top{temp}_precision"] += (cnt+len(seed_indices)) / (topk+len(seed_indices))
            total_hit[f"top{temp}_recall"] += (cnt+len(seed_indices)) / (len(seed_indices)+len(positive_indices))

        start_idx += batch_num_nodes[k]

    return total_hit

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1, help='device id')
    parser.add_argument('--input_dir', type=str, default='data', help='input directory')
    parser.add_argument('--embedding_dir', type=str, default='data', help='embedding directory')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args = parser.parse_args()

    global device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    # 加载数据集
    # xml_files = read_xml_dataset(args.input_dir)
    # data_builder = ExpandGraphDataset(xml_files=xml_files, embedding_dir=args.embedding_dir, embedding_model='BgeEmbedding', device=device, debug=args.debug)
    # 切分数据集
    # train_dataset, valid_dataset, test_dataset = split_dataset(data_builder)
    # write the dataset to disk
    # torch.save(train_dataset, 'train_dataset.pt')
    # torch.save(valid_dataset, 'valid_dataset.pt')
    # torch.save(test_dataset, 'test_dataset.pt')

    # read the dataset from disk
    # train_dataset = torch.load('train_dataset.pt')
    # valid_dataset = torch.load('valid_dataset.pt')
    test_dataset = torch.load('test_dataset.pt')

    # 使用 DataLoader 加载子集
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)
    
    logger.info(f"Load test dataset: {len(test_dataset)}")
    hit_rate = {
        'top1_hit': 0,
        'top2_hit': 0,
        'top3_hit': 0,
        'top4_hit': 0,
        'top5_hit': 0,
        'top1_precision': 0,
        'top2_precision': 0,
        'top3_precision': 0,
        'top4_precision': 0,
        'top5_precision': 0,
        'top1_recall': 0,
        'top2_recall': 0,
        'top3_recall': 0,
        'top4_recall': 0,
        'top5_recall': 0,
    }

    for i, batch_graphs in tqdm(enumerate(test_loader)):
        metrics = compute_metrics(batch_graphs.ndata['feat'], batch_graphs.ndata['label'], batch_graphs.batch_num_nodes().tolist())
        # logger.info(f"Test Batch {i}: Metrics {metrics}")
        hit_rate = {k: hit_rate[k] + metrics[k] for k in metrics}
        
    hit_rate = {k: v / len(test_loader) for k, v in hit_rate.items()}
    logger.info(f"Test finished, Test Metrics {hit_rate}")