import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import random
import time
import argparse
import torch
import logging
import dgl
import wandb
import atexit
import numpy as np

from tqdm import tqdm
from torch import nn
from itertools import product
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from code_context_model.build_dataset import ExpandGraphDataset, split_dataset
from code_context_model.gnn import RGCN, GCN, GAT, GraphSage

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# handle the exit event
def exit_handler():
    # wandb finish 
    wandb.finish()
atexit.register(exit_handler)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 计算相似度（例如使用余弦相似度）
def cosine_similarity(x1, x2):
    # 确保输入是1D张量，如果输入是2D或更高维度的张量，可以根据实际需求调整
    if x1.dim() > 1:
        x1 = x1.view(-1)
    if x2.dim() > 1:
        x2 = x2.view(-1)
    return F.cosine_similarity(x1.unsqueeze(0), x2.unsqueeze(0))

def pairwise_cosine_similarity(x1, x2):
    similarities = []
    for i in range(x1.size(0)):
        similarity = cosine_similarity(x1[i], x2[i])
        similarities.append(similarity)
    return torch.stack(similarities)
    
def euclidean_distance(x1, x2):
    # 确保输入是1D张量，如果输入是2D或更高维度的张量，可以根据实际需求调整
    if x1.dim() > 1:
        x1 = x1.view(-1)
    if x2.dim() > 1:
        x2 = x2.view(-1)
    # 计算欧式距离
    return torch.dist(x1, x2).item()

def compute_mrr(non_seed_indices, labels, similarities):
    mrr = 0
    topk = min(100, len(non_seed_indices))
    topk_indices = torch.topk(similarities, topk).indices
    for i, item in enumerate(topk_indices):
        idx = non_seed_indices[topk_indices[i]]
        if labels[idx] == 1:
            mrr = 1 / (i + 1)
            return {'MRR': mrr}
    return {'MRR': 0}

def compute_map(non_seed_indices, labels, similarities, step, topk=5):
    MAP = 0
    positive_cnt = 0
    topk = min(topk, len(non_seed_indices))
    topk_indices = torch.topk(similarities, topk).indices

    for i, item in enumerate(topk_indices):
        idx = non_seed_indices[topk_indices[i]]
        if labels[idx] == 1:
            positive_cnt += 1
            MAP += positive_cnt / (i + 1)
    MAP = MAP / step
    return {'MAP': MAP}

def compute_metrics(batch_logits, batch_labels, batch_num_nodes):
    batch_size = len(batch_num_nodes)
    start_idx = 0
    total_hit = {}
    for i in range(1, 6):
        total_hit[f'top{i}_hit'] = 0
    total_hit['mrr'] = 0
    total_hit['map'] = 0
        # total_hit[f'top{i}_precision'] = 0
        # total_hit[f'top{i}_recall'] = 0

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

        similarities = F.cosine_similarity(seed_embeddings[:,None,:] , non_seed_embeddings[None,:,:] , dim=-1) # [num_seed, num_non_seed]
        logger.debug(f"Similarities shape: {similarities.shape}")
        logger.debug(f"Similarities: {similarities}")
        # Sum the similarities across all seed embeddings
        similarities = similarities.sum(dim=0).squeeze(1)
        logger.debug(f"Summed similarities shape: {similarities.shape}")
        logger.debug(f"Similarities: {similarities}")
        # find top3 similar embeddings
        for topk in range(1,6):
            temp = topk
            topk = min(topk, len(non_seed_indices))
            topk_indices = torch.topk(similarities, topk).indices
            logger.debug(f"Top{topk} indices: {topk_indices}")
            hit = 0
            for i in range(0, len(topk_indices)):
                idx = non_seed_indices[topk_indices[i]] # get the index of the top3 embeddings
                if labels[idx] == 1:
                    hit += 1
                    break
            # total_hit[f"top{temp}_precision"] += (hit+len(seed_indices)) / (topk+len(seed_indices))
            # total_hit[f"top{temp}_recall"] += (hit+len(seed_indices)) / (len(seed_indices)+len(positive_indices))
            total_hit[f"top{temp}_hit"] += 1 if hit > 0 else 0

        mrr = compute_mrr(non_seed_indices, labels, similarities)

        step = len(positive_indices) if len(positive_indices) > 0 else 2
        map_metrics = compute_map(non_seed_indices, labels, similarities, step)

        total_hit['mrr'] += mrr['MRR']
        total_hit['map'] += map_metrics['MAP']
        start_idx += batch_num_nodes[k]

    return total_hit
    # prec = prec_metrics(logits, labels)
    # recall = recall_metrics(logits, labels)
    # f1 = f1_metrics(logits, labels)
    # return {"precision": prec.item(), "recall": recall.item(), "f1": f1.item()}

def compute_loss(batch_logits, batch_labels, batch_num_nodes, pos_margin=1.0, neg_margin=0.0):
    batch_size = len(batch_num_nodes)
    start_idx = 0
    total_losses = []
    logger.debug(f"Batch num: {batch_num_nodes}")

    for i in range(batch_size):
        # loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        labels = batch_labels[start_idx : start_idx + batch_num_nodes[i]]
        logits = batch_logits[start_idx : start_idx + batch_num_nodes[i]]
        seed_indices = (labels == -1).nonzero().view(-1)
        positive_indices = (labels == 1).nonzero().view(-1)
        negative_indices = (labels == 0).nonzero().view(-1)
        embeddings = logits
        logger.debug(f"Seed Indices: {len(seed_indices)}, Positive Indices: {len(positive_indices)}, Negative Indices: {len(negative_indices)}")
        # use predefined margin
        # neg_margin = 0.0
        # pos_margin = 1.0
        if len(positive_indices) > 0 and len(seed_indices)>0:
            # 生成所有可能的 (seed_index, positive_index) 组合对
            seed_positive_pairs = torch.cartesian_prod(seed_indices, positive_indices)
            # 提取组合对的嵌入
            seed_pair_embeddings = embeddings[seed_positive_pairs[:, 0]]
            positive_pair_embeddings = embeddings[seed_positive_pairs[:, 1]]
            # 计算正样本对之间的欧氏距离
            # positive_distances = torch.nn.functional.pairwise_distance(seed_pair_embeddings, positive_pair_embeddings)
            positive_distances = pairwise_cosine_similarity(seed_pair_embeddings, positive_pair_embeddings)
            # 构建正样本对标签（全1）
            positive_labels = torch.ones(positive_distances.size(), device=positive_distances.device)
            # 计算正样本对的 Contrastive Loss
            positive_loss = torch.mean(positive_labels *  torch.clamp(pos_margin-positive_distances, min=0.0).pow(2)) 
        else:
            positive_loss = torch.tensor(0.0, device=logits.device)

        if len(negative_indices) > 0 and len(seed_indices)>0:
            # 生成所有可能的 (seed, negative) 组合对
            seed_negative_pairs = torch.cartesian_prod(seed_indices, negative_indices)
            # 提取组合对的嵌入
            seed_pair_embeddings = embeddings[seed_negative_pairs[:, 0]]
            negative_pair_embeddings = embeddings[seed_negative_pairs[:, 1]]
            # 计算负样本对之间的欧氏距离
            # negative_distances = torch.nn.functional.pairwise_distance(seed_pair_embeddings, negative_pair_embeddings)
            negative_distances = pairwise_cosine_similarity(seed_pair_embeddings, negative_pair_embeddings)
            # 构建负样本对标签（全0）
            negative_labels = torch.zeros(negative_distances.size(), device=negative_distances.device)
            # 计算负样本对的 Contrastive Loss
            negative_loss = torch.mean((1 - negative_labels) * torch.clamp(negative_distances-neg_margin, min=0.0).pow(2))
        else:
            negative_loss = torch.tensor(0.0, device=logits.device)
        # 总的 Contrastive Loss
        total_losses.append(positive_loss + negative_loss)
        start_idx += batch_num_nodes[i]

    return torch.stack(total_losses).mean() # , non_seed_logits, non_seed_labels

def train(model: RGCN, train_loader, valid_loader, verbose=True, **kwargs):
    logger.info("======= Start training =======")
    lr = kwargs.get('lr', 0.01)
    num_epochs = kwargs.get('num_epochs', 50)
    threshold = kwargs.get('threshold', 0.5)
    output_dir = kwargs.get('output_dir', 'output')
    debug = kwargs.get('debug', False)
    pos_margin = kwargs.get('pos_margin', 1.0)
    neg_margin = kwargs.get('neg_margin', 0.0)
    # 定义损失函数和优化器
    # loss_fn = nn.BCELoss()
    # loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001, betas=(0.9, 0.99))
    
    prec_metrics = BinaryPrecision(threshold=threshold).to(device)
    recall_metrics = BinaryRecall(threshold=threshold).to(device)
    f1_metrics = BinaryF1Score(threshold=threshold).to(device)

    for epoch in tqdm(range(num_epochs)):
        total_loss, eval_loss = 0.0, 0.0
        train_graph_num_cnt = 0
        # train_avg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        # eval_avg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        train_hit_rate = {}
        eval_hit_rate = {}
        for i in range(1, 6):
            train_hit_rate[f'top{i}_hit'] = 0
            # train_hit_rate[f'top{i}_precision'] = 0
            # train_hit_rate[f'top{i}_recall'] = 0
            eval_hit_rate[f'top{i}_hit'] = 0
            # eval_hit_rate[f'top{i}_precision'] = 0
            # eval_hit_rate[f'top{i}_recall'] = 0
        train_hit_rate['mrr'] = 0
        train_hit_rate['map'] = 0
        eval_hit_rate['mrr'] = 0
        eval_hit_rate['map'] = 0

        model.train()    
        for i, batch_graphs in enumerate(train_loader):
            train_graph_num_cnt += len(batch_graphs.batch_num_nodes())
            # 打印形状以调试
            # logger.info(f"Node features shape: {batch_graphs.edata['label'].shape}")
            # logger.info(f"Node features shape: {batch_graphs.edata['label'].squeeze(1).shape}")

            # logger.info(f"Edge labels shape: {batch_graphs.edata['label'].shape}")
            batch_graphs = batch_graphs.to(device)
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['label'] = batch_graphs.edata['label'].to(device)
            batch_graphs.ndata['label'] = batch_graphs.ndata['label'].to(device)
            # logger.info("Model Forwarding...")
            logits = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['label'].squeeze(1))
            # if epoch > 10:
            #     logger.info(f"Logits: {logits}")
            #     logger.info(f"Labels: {batch_graphs.ndata['label']}")
            # loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
            # logger.info("Loss computing...")
            loss = compute_loss(logits, batch_graphs.ndata['label'], batch_graphs.batch_num_nodes().tolist(), pos_margin=pos_margin, neg_margin=neg_margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # logger.info("Metrics computing...")
            metrics = compute_metrics(logits, batch_graphs.ndata['label'], batch_graphs.batch_num_nodes().tolist()) # FIXME: wrong train metrics if batchsize > 1
            train_hit_rate = {k: train_hit_rate[k] + metrics[k] for k in metrics}
            if verbose:
                logger.info(f"Train Epoch {epoch}-Batch {i}: Loss {loss.item()}, Metrics {metrics}")
        
        # evaluate
        model.eval()
        with torch.no_grad():
            eval_graph_num_cnt = 0
            for i, batch_graphs in enumerate(valid_loader):
                batch_graphs = batch_graphs.to(device)
                batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
                batch_graphs.edata['label'] = batch_graphs.edata['label'].to(device)
                batch_graphs.ndata['label'] = batch_graphs.ndata['label'].to(device)
                eval_graph_num_cnt += len(batch_graphs.batch_num_nodes())
                logits = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['label'].squeeze(1))
                loss = compute_loss(logits, batch_graphs.ndata['label'], batch_graphs.batch_num_nodes().tolist(), pos_margin=pos_margin, neg_margin=neg_margin)
                eval_loss += loss.item()
                metrics = compute_metrics(logits, batch_graphs.ndata['label'], batch_graphs.batch_num_nodes().tolist())
                eval_hit_rate = {k: eval_hit_rate[k] + metrics[k] for k in metrics}
                if verbose:
                    logger.info(f"Valid Epoch {epoch}-Batch {i}: Loss {loss.item()}, Metrics {metrics}")
        
        train_hit_rate = {"train_"+k: v / train_graph_num_cnt for k, v in train_hit_rate.items()}
        eval_hit_rate = {"eval_"+k: v / eval_graph_num_cnt for k, v in eval_hit_rate.items()}
        wandb_log = {
            "Epoch": epoch,
            "Train Loss": total_loss,
            "Eval Loss": eval_loss,
        }
        wandb_log.update(train_hit_rate)
        wandb_log.update(eval_hit_rate)
        if not debug:
            wandb.log(wandb_log)
        logger.info(f"Epoch {epoch}, Train Loss {total_loss}, Train Metrics {train_hit_rate}")
        logger.info(f"Epoch {epoch}, Eval  Loss {eval_loss}, Eval Metrics {eval_hit_rate}")
        # save the model
        torch.save(model.state_dict(), f"{output_dir}/model_{epoch}.pth")
        logger.info(f"Model saved at {output_dir}/model_{epoch}.pth")

    logger.info("======= Training finished =======")

def test(model, test_loader, **kwargs):
    logger.info("======= Start testing =======")
    threshold = kwargs.get('threshold', 0.5)

    model.eval()
    with torch.no_grad():
        test_hit_rate = {}
        for i in range(1, 6):
            test_hit_rate[f'top{i}_hit'] = 0
            test_hit_rate[f'top{i}_precision'] = 0
            test_hit_rate[f'top{i}_recall'] = 0
        test_hit_rate['mrr'] = 0
        test_hit_rate['map'] = 0
        for i, batch_graphs in enumerate(test_loader):
            batch_graphs = batch_graphs.to(device)
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['label'] = batch_graphs.edata['label'].to(device)
            batch_graphs.ndata['label'] = batch_graphs.ndata['label'].to(device)
            logits = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['label'].squeeze(1))
            # non_seed_indices = (batch_graphs.ndata['label'] != -1).nonzero()
            # # 根据non_seed_indices选出对应的logits和labels
            # non_seed_logits = logits.squeeze(1)[non_seed_indices]
            # non_seed_labels = batch_graphs.ndata['label'].float()[non_seed_indices]
            metrics = compute_metrics(logits, batch_graphs.ndata['label'], batch_graphs.batch_num_nodes().tolist())
            # logger.info(f"Test Batch {i}: Metrics {metrics}")
            test_hit_rate = {k: test_hit_rate[k] + metrics[k] for k in metrics}
        
        test_hit_rate = {k: v / len(test_loader) for k, v in test_hit_rate.items()}
        logger.info(f"Test finished, Test Metrics {test_hit_rate}")
    

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train args
    parser.add_argument('--do_train', action='store_true', help='train the model')
    parser.add_argument('--device', type=int, default=1, help='device id')
    parser.add_argument('--input_dirs', type=str, nargs='+', default='data', help='input directories')
    parser.add_argument('--embedding_dir', type=str, default='data', help='embedding directory')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--valid_batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary classification')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model', type=str, default='rgcn', help='training model')

    # gnn layers
    parser.add_argument('--gnn_layers', type=int, default=3, help='number of gnn layers')
    parser.add_argument('--neg_margin', type=float, default=0.0, help='negative margin')
    parser.add_argument('--pos_margin', type=float, default=1.0, help='positive margin')
    # test args
    parser.add_argument('--do_test', action='store_true', help='test the model')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
    parser.add_argument('--test_model_pth', type=str, default='model_48.pth', help='test model path')

    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, f"{time.strftime('%m-%d-%H-%M')}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        args.num_epochs = 1 
        args.test_model_pth = 'model_0.pth'
    elif args.do_train:
        wandb.init(project="code-context-model")
        # 配置wandb
        config = wandb.config
        config.learning_rate = args.lr
        config.batch_size = args.train_batch_size
        config.epochs = args.num_epochs
        config.device = args.device
    if args.do_train:
        args.test_model_pth = os.path.join(args.output_dir, args.test_model_pth)   
    

    args_dict_str = '\n'.join([f"{k}: {v}" for k, v in vars(args).items()])
    logger.info(f"Arguments: \n{args_dict_str}")

    global device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    set_seed(args.seed)
    # 加载数据集
    # xml_files = read_xml_dataset(args.input_dir)
    # data_builder = ExpandGraphDataset(xml_files=xml_files, embedding_dir=args.embedding_dir, embedding_model='BgeEmbedding', device=device, debug=args.debug)
    # # 切分数据集
    # train_dataset, valid_dataset, test_dataset = split_dataset(data_builder)
    train_dataset = torch.load(os.path.join(args.input_dirs[0], 'train_dataset.pt'))
    valid_dataset = torch.load(os.path.join(args.input_dirs[0], 'valid_dataset.pt'))
    test_dataset = torch.load(os.path.join(args.input_dirs[0], 'test_dataset.pt'))
    # test_dataset = valid_dataset + train_dataset

    for i in range(1, len(args.input_dirs)):
        train_dataset = train_dataset + torch.load(os.path.join(args.input_dirs[i], 'train_dataset.pt'))
        valid_dataset = valid_dataset + torch.load(os.path.join(args.input_dirs[i], 'valid_dataset.pt'))
        test_dataset = test_dataset + torch.load(os.path.join(args.input_dirs[i], 'test_dataset.pt'))

    if args.debug:
        train_dataset = [train_dataset[i] for i in list(range(64))]  
        valid_dataset = [valid_dataset[i] for i in list(range(16))] 
        test_dataset = [test_dataset[i] for i in list(range(16))] 

    # 使用 DataLoader 加载子集
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=dgl.batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, collate_fn=dgl.batch)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, collate_fn=dgl.batch)
    logger.info(f"Load dataset finished, Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    # # 定义模型
    num_layers = 1024 # bge embedding dims
    if 'codebert' in args.embedding_dir:
        num_layers = 768
    if args.model == 'rgcn':
        model = RGCN(in_feat=num_layers, h_feat=num_layers, gnn_layers=args.gnn_layers, out_feat=1, num_rels=8)
    elif args.model == 'gat':
        model = GAT(in_feat=num_layers, h_feat=num_layers, out_feat=1, num_heads=4)
    elif args.model == 'gcn':
        model = GCN(in_feat=num_layers, h_feat=num_layers, out_feat=1)
    elif args.model == 'graphsage':
        model = GraphSage(in_feat=num_layers, h_feat=num_layers, out_feat=1)
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    model.to(device)

    if args.do_train:
        train(
            model=model, 
            train_loader=train_loader, 
            valid_loader=valid_loader, 
            verbose=False, 
            lr=args.lr,
            num_epochs=args.num_epochs,
            threshold=args.threshold,
            output_dir=args.output_dir,
            pos_margin=args.pos_margin, neg_margin=args.neg_margin,
            debug=args.debug
        )

    if args.do_test:
        logger.info(f"test model path: {args.test_model_pth}")
        model.load_state_dict(torch.load(args.test_model_pth))
        test(
            model=model, 
            test_loader=test_loader, 
            threshold=args.threshold
        )
    

    # python train.py --input_dir "" --do_train --do_test --output_dir "" --num_epochs 50 --lr 1e-4 --threshold 0.5 --seed 42